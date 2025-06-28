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

def process_single_crawl_page(cp, discovered_schema, tokenizer, device, model, llm_semaphore, relative_urls_list, stop_event):
    """
    Process a single crawl page:
    1. Visit the page (no semaphore needed)
    2. Extract URLs with context 
    3. Use semaphore for LLM relevance checking
    4. Stop when we have enough URLs
    """
    if stop_event.is_set():
        return {"processed": 0, "found": 0}
    try:
        # Step 1: Visit page (parallel, no limits)
        print(f"Crawling: {cp['url']}")
        response = wb.visit_website(cp["url"])
        if not response.get("soup"):
            return {"processed": 0, "found": 0}
        # Step 2: Extract URLs with context (parallel, no limits)
        urls = wb.find_url_with_context(response)
        if not urls:
            return {"processed": 0, "found": 0}
        found_count = 0
        # Step 3: Process URLs in batches with LLM semaphore
        for num in range(0, len(urls), 5):
            if stop_event.is_set():
                break
            batch = urls[num:num+5]
            # Acquire semaphore for LLM call (limited concurrent access)
            with llm_semaphore:
                if stop_event.is_set():
                    break
                try:
                    gen_urls = intg.check_relevance(batch, discovered_schema, tokenizer, device, model)
                except Exception as e:
                    print(f"Error in relevance check: {e}")
                    try:
                        # Fallback: try with remaining URLs
                        gen_urls = intg.check_relevance(urls[num:], discovered_schema, tokenizer, device, model)
                    except:
                        continue
                if gen_urls and gen_urls.get('probable_urls'):
                    # Thread-safe addition to shared list
                    with threading.Lock():
                        if not stop_event.is_set():
                            relative_urls_list.extend(gen_urls['probable_urls'])
                            found_count += len(gen_urls['probable_urls'])
                            print(f"Found {len(gen_urls['probable_urls'])} relevant URLs. Total: {len(relative_urls_list)}/20")
                            print(f"Reasoning: {gen_urls.get('reasoning', 'No reasoning provided')}")
                            # Stop when we have enough
                            if len(relative_urls_list) >= 20:
                                print("Reached target of 20 relevant URLs, stopping all threads")
                                stop_event.set()
                                break
        return {"processed": len(urls), "found": found_count}
    except Exception as e:
        print(f"Error processing {cp['url']}: {e}")
        return {"processed": 0, "found": 0}

def parallel_relative_pages_crawl(crawl_pages, discovered_schema, tokenizer, device, model):
    """
    Parallelize the crawling with controlled LLM access
    """
    # Shared data structures
    relative_urls_list = []
    stop_event = threading.Event()
    # Semaphore to limit concurrent LLM calls (only 2-3 at a time)
    llm_semaphore = threading.Semaphore(2)  # Adjust based on your GPU memory
    # Thread lock for shared list access
    results_lock = threading.Lock()
    # Use more workers for crawling since most time is I/O bound
    max_crawl_workers = min(8, len(crawl_pages))
    print(f"Starting parallel crawl with {max_crawl_workers} workers and LLM semaphore of 2")
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=max_crawl_workers) as executor:
        # Submit all crawling tasks
        future_to_page = {
            executor.submit(
                process_single_crawl_page, 
                cp, discovered_schema, tokenizer, device, model, 
                llm_semaphore, relative_urls_list, stop_event
            ): cp for cp in crawl_pages
        }
        # Monitor completion
        completed = 0
        total_processed = 0
        total_found = 0
        for future in as_completed(future_to_page):
            if stop_event.is_set():
                # Cancel remaining futures
                for f in future_to_page:
                    if not f.done():
                        f.cancel()
                break
            try:
                result = future.result()
                total_processed += result["processed"]
                total_found += result["found"]
                completed += 1
                if completed % 5 == 0:  # Progress update every 5 completions
                    print(f"Progress: {completed}/{len(crawl_pages)} pages crawled, {len(relative_urls_list)} relevant URLs found")
            except Exception as e:
                page = future_to_page[future]
                print(f"Exception in thread for {page['url']}: {e}")
    end_time = time.time()
    print(f"Parallel crawl completed in {end_time - start_time:.2f} seconds")
    print(f"Processed {total_processed} URLs across {completed} pages")
    print(f"Found {len(relative_urls_list)} relevant URLs")
    return relative_urls_list

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
    model = llm_context["model"]
    tokenizer = llm_context["tokenizer"]
    device = llm_context["device"]
    
    if not state["discovered_schema"]:
        state["relative_pages"] = {"error": "No schema discovered"}
        return state
    
    # 1. Get all URLs from already scraped pages (keep this part as is)
    crawl_pages = []
    for url in state['search_results']:
        print(f"Extracting URLs from: {url['title']} | {url['url']}")
        page = wb.visit_website(url['url'])
        if not page.get('error'):
            if len(crawl_pages) == 0:
                crawl_pages = wb.get_all_urls(page['soup'])
            else:
                crawl_pages.extend(wb.get_all_urls(page['soup']))
        else:
            print(f"Error {page['error']} for {url['title']} | {url['url']}")
    
    if not crawl_pages:
        state['relative_urls'] = []
        return state
    
    print(f"Found {len(crawl_pages)} URLs to crawl")
    
    # 2. Parallel crawling with semaphore-controlled LLM calls
    relative_urls = parallel_relative_pages_crawl(
        crawl_pages, state["discovered_schema"], tokenizer, device, model
    )
    
    state['relative_urls'] = relative_urls
    print(f"Final result: Found {len(relative_urls)} relevant URLs")
    for idx , rel in enumerate(relative_urls):
        print(f"No. {idx} :\n{rel}")
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
    graph.add_node("relative_pages",relative_pages)

    # Add edges
    graph.add_edge("initialize_system", "query_expansion")
    graph.add_edge("query_expansion", "web_search")
    #graph.add_edge("web_search","select_seeds")
    graph.add_edge("web_search","generate_schema")
    graph.add_edge("generate_schema","relative_pages")
    
    #graph.add_edge("select_seeds", END)
    graph.add_edge("relative_pages", END)
    
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