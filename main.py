from langgraph.graph import END, StateGraph
from typing import Dict, List, Any, TypedDict, Optional
import pandas as pd
import json
import time

# Import your existing modules
import intelligence as intg
import web_search as wb

# State definition
class ScraperState(TypedDict, total=False):
    # Required fields
    query: str
    search_results: List[Dict]
    fetched_pages: List[Dict]
    current_batch: List[Dict]
    examine_urls: List[str]
    relative_urls: List[Dict]
    discovered_schema: Dict
    batch_size: int
    batch_count: int
    # Optional fields
    dataset_path: Optional[str]
    selected_column: Optional[str]
    dataset_entities: List
    error: Optional[str]
    # a context field to pass non-serializable objects like the model
    context: Dict[str, Any]

# Node functions
def initialize_system(state: ScraperState) -> ScraperState:
    """Initialize the system and load dataset if provided"""
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

def generate_search_queries(state: ScraperState) -> ScraperState:
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
                query_num=2
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
    # Now search for these queries
    search_results = []
    if queries:
        search_results = wb.duckduckgo_search_requests(queries)# this has to be a separate state ?
        print(f"Found {len(search_results)} search results")
    
    state["search_results"] = search_results
    return state 

def fetch_batch_of_pages(state: ScraperState) -> Dict:
    """
    Fetch a batch of web pages from search results
        1. visit pages .
        2. get paragraphs .
    """
    batch_size = state["batch_size"]
    start_idx = state["batch_count"] * batch_size
    
    # Filter out URLs we've already processed
    already_fetched_urls = {page["url"] for page in state["fetched_pages"]}
    remaining_results = [r for r in state["search_results"] 
                         if r["url"] not in already_fetched_urls]
    
    if not remaining_results:
        print("No more results to process")
        return {"state": state, "next": "generate_schema"}
    
    # Get current batch
    end_idx = min(start_idx + batch_size, len(remaining_results))
    current_batch = remaining_results[start_idx:end_idx]
    
    print(f"Fetching batch {state['batch_count']+1}: {len(current_batch)} pages")
    fetched_batch = []
    
    for result in current_batch:
        print(f"Visiting page: {result['title']}")
        try:
            # Use your visit_website function
            response = wb.visit_website(result["url"])
            if response.get("soup"):
                # Extract paragraphs
                paragraphs = wb.get_all_paragraphs(response)
                # Store the content
                fetched_batch.append({
                    "url": result["url"],
                    "title": result["title"],
                    "content": "\n".join([p["text"] for p in paragraphs])
                })
                
                time.sleep(1)  # Be polite to servers
        except Exception as e:
            print(f"Error fetching {result['url']}: {e}")
    
    # Update state
    state["current_batch"] = fetched_batch
    state["fetched_pages"].extend(fetched_batch)
    state["batch_count"] += 1
    
    # Decide next action
    if len(fetched_batch) == 0:
        print("Empty batch, moving to schema generation")
        return {"state": state, "next": "generate_schema"}
    elif state["batch_count"] >= 5:
        print("Reached maximum batch count, moving to schema generation")
        return {"state": state, "next": "generate_schema"}

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
            print(f"{urls}")
            # give urls in batches
            prevTake=0# it has to stop sometime...
            toBreak = 0
            for num in range(0,len(urls),10):
                try :
                    gen_urls = intg.check_relevance(urls[num:num+10],state["discovered_schema"], tokenizer , device , model)
                except Exception as e:
                    gen_urls = intg.check_relevance(urls[num:],state["discovered_schema"], tokenizer , device , model)
                finally :
                    if gen_urls:
                        # this takes a lot of time . 
                        # it is also inaccurate .
                        # let's simplify .
                        if gen_urls.get('probable_urls',None):
                            relative_urls.extend([gen_urls])
                            if len(gen_urls['probable_urls'])>=2 or len(relative_urls)>25:#hardcoded for now
                                break
                        #    prevTake=len(gen_urls['probable_urls'])
                        #else:
                        #    if toBreak>1:
                        #    toBreak+=1
                    #endif
                print(relative_urls)
        #endif
        if len(relative_urls)>25:
            print("Found enough relative pages, stopping search")
            break
    #endfor
    state['relative_urls'] = relative_urls

def create_dataset(state: ScraperState) -> ScraperState:
    global llm_context
    # Access the global context
    global llm_context
    model = llm_context["model"]
    tokenizer = llm_context["tokenizer"]
    device = llm_context["device"]
    schema = state["discovered_schema"]
    if not state["relative_urls"]:
        state["discovered_schema"] = {"error": "No schema was produced ."}
        return state
    '''
        Crawl probable webpages based on previous results .
        1. get urls from already scrapped pages .
        2. find relevance
        3. populate
    '''
    # 1. get all urls
    crawl_pages = []
    for urls in state['probable_urls']:
        for url in urls:
            # 1.1 get urls from pages
            page = wb.visit_website(url)
            par = wb.get_all_paragraphs(page['soup'])
            text = "\n".join([p["text"] for p in par])
            row = intg.row_creation(text , schema , tokenizer , device , model)
            print(f"Row created: {row}")
    #endfor
   
# Define a global context variable to store non-serializable objects
llm_context = {}

# Define the graph
def create_scraper_graph():
    '''
        A function to initialize the workflow
    '''
    graph = StateGraph(ScraperState)
    graph.add_node("initialize_system", initialize_system)
    graph.add_node("generate_search_queries", generate_search_queries)# create web queries and find pages .
    # improve
    graph.add_node("fetch_batch_of_pages", fetch_batch_of_pages)# fetch pages in batches to determine the probable schema .
    graph.add_node("generate_schema", generate_schema)
    graph.add_node("relative_pages" , relative_pages)
    graph.add_node("create_dataset", create_dataset)  # Create dataset from the schema
    # start crawling
    
    # Add edges
    graph.add_edge("initialize_system", "generate_search_queries")
    graph.add_edge("generate_search_queries", "fetch_batch_of_pages")
    graph.add_edge("fetch_batch_of_pages", "generate_schema")
    graph.add_edge("generate_schema", "relative_pages")
    graph.add_edge("relative_pages", "create_dataset")

    graph.add_edge("create_dataset", END)
    
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
        "search_results": [],
        "fetched_pages": [],
        "current_batch": [],
        "discovered_schema": {},
        "relative_urls": [],
        "batch_size": 5,
        "batch_count": 0,
        "dataset_path": dataset_path,
        "selected_column": selected_column,
        "dataset_entities": [],
        "examine_urls" : []
    }
    
    # Run the graph
    result = scraper.invoke(state)
    
    # Show results
    print("\n\nFinal Results:")
    print(f"Pages fetched: {len(result['fetched_pages'])}")
    print("\nDiscovered Schema:")
    print(json.dumps(result["discovered_schema"], indent=2))
    print("\nSchema saved to 'discovered_schema.json'")

if __name__ == "__main__":
    main()