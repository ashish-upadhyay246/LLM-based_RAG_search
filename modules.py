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
warnings.filterwarnings('ignore') 

load_dotenv()

genai.configure(api_key=os.getenv('GEMINI_API_KEY'))
MODEL_ID = "models/text-embedding-004"

link_count=0

#function to crawl the search results and scrape the sites
def googlebot(url, sites_required):
    global link_count
    scraped_text=""
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.93 Safari/537.36'}
    extracted_links={}
    sites_to_scrape=sites_required+30
    page_count=0

    #extracting the search links pagewise.
    while(len(extracted_links)<sites_to_scrape):
        page_temp=page_count*10
        if(page_count>1):
            url_temp=url[0:-2]
        else:
            url_temp=url
        url=url_temp+str(page_temp)

        url_open = requests.get(url, headers=headers)
        url_open.raise_for_status()
        soup = BeautifulSoup(url_open.content, 'html5lib')
        for result in soup.find_all('div', class_='g'):
            heading = result.find('h3')
            link = result.find('a', href=True)
            if heading and link:
                text = heading.get_text(strip=True)
                text_url = link['href']
                extracted_links[link_count + 1] = (text, text_url)
                link_count += 1
        page_count += 1
         

    print("\nExtracted links: ")
    for x, (y,l) in extracted_links.items():
        print(f"{x}. {y} - {l}")
    print("\n")

    #scraping the required number of links.
    mx=1
    sites=0
    i=1
    while i<sites_required+mx:
        link=extracted_links[i][1]
        if '/maps' in link or '/search?' in link or 'www.instagram' in link or '/shop/' in link or "myntra" in link or "play.google" in link or "flipkart.com" in link or "amazon.in" in link:
            print("Error scraping site no.",i, " Reason: found maps/search?/instagram")
            mx+=1
            i+=1
            continue
        try:
            url_open2=requests.get(link, timeout=20)
        except requests.exceptions.RequestException as e:
            print("Error scraping site no.", i)
            mx+=1
            i+=1
            continue
        sites+=1
        soup2=BeautifulSoup(url_open2.content, 'html5lib')
        for j in soup2.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6']):
            if j.name=='a' or j.name=='img':
                continue
            text=j.get_text(strip=False)
            if text:
                scraped_text+=text
        print("Scraped: ",i," TOTAL SCRAPED SITES: ", sites)
        i+=1
    print("TOTAL SCRAPED SITES: ", sites)
    
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
    no_of_relevant_chunks = max(20,sites_required*2)
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
def generateResponse(query, relevant_chunks):
    print("Formulating response.")
    model = genai.GenerativeModel(model_name="gemini-1.5-flash")
    
    print("\nLength of relevant chunks: ")
    print(len(relevant_chunks))
    
    context = " ".join(relevant_chunks)
    prompt = f"The query given by the user follows after the colon: {query}\n"
    rules = "Strictly use the following text as the database only to generate responses for the previous query. Strictly do not use your own database or knowledge about the query. The response length should be directly proportional to the number of relevant chunks. The text will follow after this colon:\n"
    search_prompt = prompt + rules + context

    response = model.generate_content(search_prompt)
    print(response)
    return response.text


def main(q, https_match, sites_required):
    global link_count
    link_count=0
    text=search_and_scrape(q,https_match, sites_required) #scrape the web
    chunks=chunk_text(text, 500) #divide into chunks
    print("Total number of chunks: ")
    print(len(chunks))
    print("Creating embeddings.")
    embeddings = parallel_embed_text_chunks(chunks) #make embeddings for the chunks
    index=store_embeds(embeddings) #indexing of embeds
    q_embedding=embed_query(q) #embedding the query
    relevant_chunks=retrieve_relevant_chunks(q_embedding, index, chunks, sites_required)
    return generateResponse(q, relevant_chunks)
