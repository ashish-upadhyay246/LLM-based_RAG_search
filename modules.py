import re
import os
import faiss
import time
import warnings
import requests
import html5lib
import numpy as np
import google.generativeai as genai

from bs4 import BeautifulSoup
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from requests.exceptions import RequestException
from langchain.memory import ConversationBufferMemory
warnings.filterwarnings('ignore') 

load_dotenv()
memory=ConversationBufferMemory(return_messages=True)
genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
MODEL_ID = "models/text-embedding-004"

link_count=0


#scraper
def scrape_site(link, site_number):
    try:
        url_open2 = requests.get(link, timeout=10)
        soup2 = BeautifulSoup(url_open2.content, 'html5lib')
        scraped_text = ""
        for j in soup2.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if j.name == 'a' or j.name == 'img':
                continue
            text = j.get_text(strip=False)
            if text:
                scraped_text += text
        print("Scraped site no. ",site_number)
        return scraped_text
    except Exception as e:
        print(f"Error scraping site no. {site_number}: {e}")
        return ""

#search links
def extract_links_from_search_page(url, page_number, headers):
    try:
        page_url = url + str(page_number * 10)
        response = requests.get(page_url, headers=headers)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html5lib')
        links = {}
        for result in soup.find_all('div', class_='g'):
            heading = result.find('h3')
            link = result.find('a', href=True)
            if heading and link:
                links[heading.get_text(strip=True)] = link['href']
        return links
    except Exception as e:
        print(f"Error fetching search page {page_number}: {e}")
        return {}


def googlebot(url, sites_required):
    global link_count
    scraped_text = ""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'}
    extracted_links = {}

    # Determine how many pages to fetch
    total_pages_to_fetch = (sites_required + 10)//10

    # Fetch and parse search results pages concurrently
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_links_from_search_page, url, i, headers) for i in range(total_pages_to_fetch)]
        for future in futures:
            links = future.result()
            for title, link in links.items():
                extracted_links[link_count + 1] = (title, link)
                link_count += 1         

    # print("\nExtracted links: ")
    # for x, (y,l) in extracted_links.items():
    #     print(f"{x}. {y} - {l}")
    # print("\n")

    #scraping concurrently
    valid_links = []
    for i in range(1, sites_required + 1):
        link = extracted_links[i][1]
        if not any(banned in link for banned in ['/maps', '/search?', 'instagram', 'shop', 'myntra', 'play.google', 'flipkart.com', 'amazon.in']):
            valid_links.append((link, i))

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda x: scrape_site(x[0], x[1]), valid_links))

    scraped_text = "".join(results)
    return scraped_text      

def urlbot(url):
    ans = ""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html5lib')
        
        for i in soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if i.name == 'a' or i.name == 'img':
                continue
            text = i.get_text(strip=False)
            if text:
                ans += text

        return ans
    except requests.exceptions.RequestException as e:
        print(f"Request error for URL {url}: {e}")
    except Exception as e:
        print(f"Error while processing URL {url}: {e}")

#check query and initiate scraping
def search_and_scrape(q,https_match, sites_required):
    print("\nInitializing search and scrape.")
    if re.match(https_match, q):
        url = q
        return urlbot(url)
    else:
        lists = q.split()
        word = "+".join(lists)
        url = "https://www.google.com/search?q=" + word + "&start="
        print("Calling web scraper.")
        return googlebot(url, sites_required)

#speeding up embeddings by use of thread management
def parallel_embed_text_chunks(chunks):
    if(len(chunks)>1480):
        chunks=chunks[:1480:]
    with ThreadPoolExecutor() as executor:
        embeddings = list(executor.map(embed_text_chunk, chunks))
    return embeddings

#creating embeddings
def embed_text_chunk(content, retries=3, backoff_factor=1):
    for i in range(retries):
        try:
            result = genai.embed_content(
                model=MODEL_ID,
                content=content,
                task_type="retrieval_document",
                title="Embedding of single string"
            )
            return np.array(result['embedding'], dtype='float32')
        except Exception as e:
            print(f"Error embedding text chunk ({e}). Retrying in {backoff_factor * (2 ** i)} seconds...")
            time.sleep(backoff_factor * (2 ** i))
    print("Max retries exceeded for embedding text chunk.")
    return np.zeros((1, 512), dtype='float32')
    
#creating embeddings for the query
def embed_query(query):
    print("\nCreating embedding for the query.")
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_document",
        title="Embedding of user query"
    )
    return result['embedding']

#finding the relevant chunks out of total chunks
def retrieve_relevant_chunks(query_embedding, index, text_chunks, sites_required):
    no_of_relevant_chunks = int(sites_required*3)
    n=int(len(text_chunks))
    if(no_of_relevant_chunks>n):
        no_of_relevant_chunks=text_chunks
    print("\nRetrieving relevant chunks.")
    query_vector = np.array([query_embedding], dtype='float32')
    _, indices = index.search(query_vector, no_of_relevant_chunks)
    relevant_chunks = [text_chunks[i] for i in indices[0]]
    return relevant_chunks

#creating chunks from the scraped text
def chunk_text(text, chunk_size):
    print("\nCreating chunks.")
    chunk_list= [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunk_list

#indexing and storing the embeddings 
def store_embeds(embeddings):
    print("\nStoring embeddings.")
    faiss_embeddings = np.array(embeddings, dtype='float32')
    dimension = faiss_embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(faiss_embeddings)
    return index

#initializing geminiAPI to generate response based on relevant chunks and the query provided.
def generateResponse(query, relevant_chunks, memory):
    print("Formulating response.")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    print("\nLength of relevant chunks: ")
    print(len(relevant_chunks))


    
    context = " ".join(relevant_chunks)
    prompt = f"The query given by the user follows after the colon: {query}\n"
    rules = "You must strictly use only the following text as the database to generate responses. Do not use your own knowledge. If the answer is not in the text, say 'The answer is not available in the provided context.' The relevant text is:\n"
    search_prompt = prompt + rules + context

    conversation_history = memory.load_memory_variables({})
    conversation_context = conversation_history.get('history', '')
    final_prompt = f"{conversation_context}\n{search_prompt}"

    response = model.generate_content(final_prompt)

    memory.save_context({"input": query}, {"output": response.text})
    return response.text

cached_chunks = None
cached_embeddings = None
cached_index = None
last_query = None

def clear_cache():
    global cached_chunks, cached_embeddings, cached_index, last_query
    cached_chunks = None
    cached_embeddings = None
    cached_index = None
    last_query = None

def main(q, https_match, sites_required):
    global link_count, cached_chunks, cached_embeddings, cached_index, last_query
    
    # Load conversation history to get previous queries
    conversation_history = memory.load_memory_variables({})
    # Access the content of HumanMessage objects correctly
    previous_queries = [msg.content for msg in conversation_history.get('history', []) if hasattr(msg, 'content')]
    
    # Concatenate previous queries with the current query
    combined_query = " ".join(previous_queries) + " " + q
    print(f"Combined Query: {combined_query}")  # For debugging

    if cached_chunks and cached_embeddings and cached_index:
        print("\nReusing cached chunks and embeddings.")
    else:
        link_count = 0
        text = search_and_scrape(q, https_match, sites_required)  # scrape the web
        cached_chunks = chunk_text(text, 500)  # divide into chunks
        print("Total number of chunks: ")
        print(len(cached_chunks))
        print("Creating embeddings.")
        cached_embeddings = parallel_embed_text_chunks(cached_chunks)  # make embeddings for the chunks
        cached_index = store_embeds(cached_embeddings)  # indexing of embeds

    q_embedding = embed_query(combined_query)  # embedding the combined query
    relevant_chunks = retrieve_relevant_chunks(q_embedding, cached_index, cached_chunks, sites_required)
    
    last_query = q
    return generateResponse(combined_query, relevant_chunks, memory)
