from langgraph.graph import END, StateGraph
from typing import Dict, List, Any, TypedDict, Optional
import pandas as pd
import json
import time

# Import your existing modules
import intelligence as intg
import web_search as wb
import nlp
# parallel
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

def fetch_single_page(result):
    """
    Fetch a single page and return the processed data
    """
    try:
        print(f"Visiting page: {result['title']} | {result['url']}")
        page = wb.visit_website(result['url'])
        
        if not page.get('error'):
            # extract text as a whole
            text = page['soup'].get_text()
            # no long spaces
            text = ' '.join(text.split())
            
            page_data = {
                "title": result['title'],
                "url": result['url'],
                "content": page['soup'].get_text(),
                "raw_text": text,
                "preprocessed_text": nlp.preprocess_text(text),
                "soup": page['soup']
            }
            print(f"Fetched {result['title']} successfully")
            return {"success": True, "data": page_data, "error": None}
        else:
            print(f"Error fetching {result['title']}: {page['error']}")
            return {"success": False, "data": None, "error": page['error']}
            
    except Exception as e:
        print(f"Exception fetching {result['title']}: {str(e)}")
        return {"success": False, "data": None, "error": str(e)}
    
# State definition
class ScraperState(TypedDict, total=False):
    # Required fields
    query: str
    queries_expansion: List[str]
    search_results: List[Dict[str, Any]]
    fetched_pages: List[Dict[str, Any]]
    discovered_schema: Dict

# Node functions
def initialize_system(state: ScraperState) -> ScraperState:
    """
        Initialize the system and load dataset if provided
    """
    print("Initializing LLM...")
    model, tokenizer, device = intg.initialize_llm()
    
    if model is None:
        state["error"] = "Failed to initialize LLM"
        return {"state": state, "next": END}
    
    # Set up a context object to store non-serializable items
    global llm_context
    llm_context = {
        "model": model,
        "tokenizer": tokenizer,
        "device": device
    }
    
    # Load dataset if path was provided
    if state.get("dataset_path"):
        try:
            df = pd.read_csv(state["dataset_path"])
            if state.get("selected_column") and state["selected_column"] in df.columns:
                state["dataset_entities"] = df[state["selected_column"]].tolist()
                print(f"Loaded {len(state['dataset_entities'])} entities from dataset")
            else:
                # No column specified, show options
                print("Available columns:")
                for i, col in enumerate(df.columns):
                    print(f"{i}. {col}")
                # This would typically be an input, but we'll use the first column by default
                state["selected_column"] = df.columns[0]
                state["dataset_entities"] = df[state["selected_column"]].tolist()
        except Exception as e:
            print(f"Error loading dataset: {e}")
            state["dataset_entities"] = []
    else:
        state["dataset_entities"] = []
    
    return state

def query_expansion(state: ScraperState) -> ScraperState:
    """Generate intelligent search queries based on the user query"""
    # Access the global context
    global llm_context
    model = llm_context["model"]
    tokenizer = llm_context["tokenizer"]
    device = llm_context["device"]
    query = state["query"]
    
    # If we have dataset entities, generate queries for each entity
    queries = []
    
    if state.get("dataset_entities"):
        # Use entities from dataset
        for entity in state["dataset_entities"][:5]:  # Limit to first 5 entities for demo
            entity_queries = intg.query_creator(
                model, tokenizer, device, 
                entity=entity, 
                context="", 
                user_prompt=query,
                query_num=4
            )
            print(f"Generated queries for {entity}: {entity_queries}")
            queries.extend(entity_queries)
    else:
        # No dataset, just use the user query
        queries = intg.query_creator(
            model, tokenizer, device, 
            entity="", 
            context="", 
            user_prompt=query,
            query_num=3
        )
        print(f"Generated queries: {queries}")
    
    state["queries_expansion"] = queries
    return state 

def web_search(state: ScraperState) -> ScraperState:
    """
    Fetch a batch of web pages from search results
        1. web search using the query_expansion results .
        2. visit the pages and store their content .
    """
    # Access the global context
    global llm_context
    model = llm_context["model"]
    tokenizer = llm_context["tokenizer"]
    device = llm_context["device"]
    queries = state["queries_expansion"]
    web_search_reasults = wb.duckduckgo_search_requests(queries)
    if not web_search_reasults:
        state["search_results"] = []
        state["error"] = "No search results found"
        return {"state": state, "next": END}
    print(f"Found {len(web_search_reasults)} search results")
    state["search_results"] = web_search_reasults
    # visit the pages and store their content
    # Parallel page fetching
    fetched_pages = []
    error_count = 0
    # Use ThreadPoolExecutor for parallel execution
    max_workers = min(8, len(state["search_results"]))  # Limit concurrent requests
    print(f"Starting parallel fetch with {max_workers} workers...")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_result = {
            executor.submit(fetch_single_page, result): result 
            for result in state["search_results"]
        }
        # Collect results as they complete
        for future in as_completed(future_to_result):
            result = future_to_result[future]
            try:
                page_result = future.result()
                if page_result["success"]:
                    fetched_pages.append(page_result["data"])
                else:
                    error_count += 1
            except Exception as e:
                print(f"Exception in thread for {result['title']}: {str(e)}")
                error_count += 1
    # end parallel thread pool
    end_time = time.time()
    print(f"Parallel fetch completed in {end_time - start_time:.2f} seconds")
    print(f"Errors occurred: {error_count}/{len(state['search_results'])}")
    if not fetched_pages:
        state["fetched_pages"] = []
        state["error"] = "No pages were successfully fetched"
        return {"state": state, "next": END}
    print(f"Fetched {len(fetched_pages)} pages successfully")
    state["fetched_pages"] = fetched_pages
    return state

def select_seeds(state: ScraperState) -> ScraperState:
    global llm_context
    # Access the global context
    global llm_context
    model = llm_context["model"]
    tokenizer = llm_context["tokenizer"]
    device = llm_context["device"]
    """
        1. Generate topic seeds using bertopic model .
    """
    if not state["fetched_pages"]:
        state["discovered_schema"] = {"error": "No pages were successfully fetched"}
        return state
    df = pd.DataFrame(state["fetched_pages"])
    print(df.head())
    topics , _ = nlp.extract_topics(df, model)
    print(f"Discovered {len(set(topics))} topics")
    for i, topic in enumerate(set(topics)):
        print(f"Topic {i}: {topic}")
    return state

def generate_schema(state: ScraperState) -> ScraperState:
    """
        Generate a schema based on the fetched content
        - fetch the content of some pages alongside a probable schema .
        - let the llm create the schema .
    """
    # Access the global context
    global llm_context
    model = llm_context["model"]
    tokenizer = llm_context["tokenizer"]
    device = llm_context["device"]
    if not state["fetched_pages"]:
        state["discovered_schema"] = {"error": "No pages were successfully fetched"}
        return state

    schema_json , generated_text = intg.infer_schema(state['fetched_pages'] , state['query'] , state['discovered_schema'] , tokenizer , device , model)
    try :
        if isinstance(schema_json, str) and schema_json.startswith("Error"):
            # Fallback if JSON extraction fails
            print("Error")
            state["discovered_schema"] = {"raw_response": generated_text}
        else:
            state["discovered_schema"] = schema_json
    except Exception as e:
        print(f"Error generating schema: {e}")
        state["discovered_schema"] = {"error": str(e)}
    
    # Save schema to file
    with open("discovered_schema.json", "w") as f:
        json.dump(state["discovered_schema"], f, indent=2)
    
    return state

def relative_pages(state: ScraperState) -> ScraperState:
    global llm_context
    # Access the global context
    global llm_context
    model = llm_context["model"]
    tokenizer = llm_context["tokenizer"]
    device = llm_context["device"]
    
    if not state["discovered_schema"]:
        state["relative_pages"] = {"error": "No pages were successfully fetched"}
        return state
    '''
        Crawl probable webpages based on previous results .
        1. get urls from already scrapped pages .
        2. find relevance
        3. populate
    '''
    # 1. get all urls
    crawl_pages = []
    for url in state['search_results']:
        # 1.1 get urls from pages
        print(f"Visiting page : {url['title']} | {url['url']}")
        page = wb.visit_website(url['url'])
        if not page.get('error') :
            if len(crawl_pages)==0:
                crawl_pages = wb.get_all_urls(page['soup'])
            else:
                crawl_pages.extend(wb.get_all_urls(page['soup']))
            #endif
        else:
            print(f"Error {page['error']} for {url['title']} | {url['url']}")
        #endif
    #endfor
    # 2. visit pages and check if relevant
    relative_urls = []
    for cp in crawl_pages :
        response = wb.visit_website(cp["url"])
        if response.get("soup"):
            # Extract paragraphs
            urls = wb.find_url_with_context(response)
            # give urls in batches
            for num in range(0,len(urls),5):
                try :
                    gen_urls = intg.check_relevance(urls[num:num+5],state["discovered_schema"], tokenizer , device , model)
                except Exception as e:
                    gen_urls = intg.check_relevance(urls[num:],state["discovered_schema"], tokenizer , device , model)
                finally :
                    if gen_urls:
                        # this takes a lot of time . 
                        # it is also inaccurate .
                        # let's simplify .
                        print(f"{gen_urls['reasoning']}")
                        print(f"\n\nFound {len(relative_urls)}/25 .\n\n")
                        if gen_urls.get('probable_urls',None):
                            relative_urls.extend(gen_urls.get('probable_urls'))
                            if len(gen_urls['probable_urls'])>=2 or len(relative_urls)>25:#hardcoded for now
                                break
                    #endif
        #endif
        if len(relative_urls)>25:
            print("Found enough relative pages, stopping search")
            break
    #endfor
    state['relative_urls'] = relative_urls
    print(f"Found {state['relative_urls']} relative URLs")
    return state

def create_dataset(state: ScraperState) -> ScraperState:
    print("Creating dataset from discovered schema...")
    print(f"Relative URLs: {state['relative_urls']}")
    global llm_context
    # Access the global context
    global llm_context
    model = llm_context["model"]
    tokenizer = llm_context["tokenizer"]
    device = llm_context["device"]
    schema = state["discovered_schema"]
    #if not state["relative_urls"]:
    #    state["discovered_schema"] = {"error": "No schema was produced ."}
    #    return state
    '''
        Crawl probable webpages based on previous results .
        1. get urls from already scrapped pages .
        2. find relevance
        3. populate
    '''
    # 1. get all urls
    crawl_pages = []
    for url in state['relative_urls']:
        # 1.1 get urls from pages
        try:
            page = wb.visit_website(url)
            par = wb.get_all_paragraphs(page['soup'])
            text = "\n".join([p["text"] for p in par])
            row = intg.row_creation(text , schema , tokenizer , device , model)
            print(f"Row created: {row}")
        except Exception as e:
            print(f"Error processing {url}: {e}")
            continue
        #endif
    #endfor
   
# Define a global context variable to store non-serializable objects
llm_context = {}

# Define the graph
def create_scraper_graph():
    '''
        A function to initialize the workflow
    '''
    graph = StateGraph(ScraperState)
    # Phase 0: Initialize the system
    print("Phase 0: Initializing the system...")
    graph.add_node("initialize_system", initialize_system)
    # Phase 1: Query Expansion
    print("Phase 1: Query Expansion...")
    graph.add_node("query_expansion", query_expansion)
    # Phase 2: Web Search
    print("Phase 2: Web Search...")
    graph.add_node("web_search", web_search)
    # Phase 3: Generate Schema
    print("Phase 3: Generate Schema...")
    #graph.add_node("select_seeds", select_seeds)
    graph.add_node("generate_schema", generate_schema)
    # Phase 4: Crawl
    print("Phase 4: Crawl...")

    # Add edges
    graph.add_edge("initialize_system", "query_expansion")
    graph.add_edge("query_expansion", "web_search")
    #graph.add_edge("web_search","select_seeds")
    graph.add_edge("web_search","generate_schema")
    
    #graph.add_edge("select_seeds", END)
    graph.add_edge("generate_schema", END)
    
    # Set entry point
    graph.set_entry_point("initialize_system")
    
    return graph.compile()

# Usage example
def main():
    # Create the graph
    scraper = create_scraper_graph()
    
    # Get user input
    user_query = input("What would you like to search for? ")
    
    # Ask if user wants to use a dataset
    use_dataset = input("Do you want to use a dataset? (y/n): ").lower().startswith('y')
    dataset_path = None
    selected_column = None
    
    if use_dataset:
        dataset_path = input("Enter dataset path: ")
        
    # Initialize state
    state = {
        "query": user_query,
        "queries_expansion": [],
        "search_results": [],
        "discovered_schema": {},
    }
    
    # Run the graph
    result = scraper.invoke(state)
    
if __name__ == "__main__":
    main()