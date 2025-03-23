import json
import asyncio
from pydantic import BaseModel, Field
import random
import time
import os

from tqdm.asyncio import tqdm_asyncio
from together import Together

# Initialize Together API client
together = Together(api_key="cc0d91ebee779df1ec459dd294cc936cdcf85458b3422b63240e196a5d2a53a3")

GENRES = ['romantik', 'fəlsəfi', 'tarixi', 'lirik', 'epik', 'satirik']

class LiteratureText(BaseModel):
    text: str = Field(description="Azərbaycan ədəbiyyatından mətnlər")

async def generate_literature_async(genres=GENRES, temperature=0.7, max_retries=3):
    """Generate a single literature sample asynchronously"""

    genre = random.choice(genres)
    
    transcript = (
        "Azərbaycanca verilən janrlara uyğun ədəbi uzun mətnlər yarat. "
        "Hər mətndə yalnız həmin janrın xüsusiyyətləri əks olunmalıdır. "
        "Mətndə göstərilməlidir:\n\nJanr: " + genre
    )
    
    system_prompt = (
        "Sən professional Azərbaycan ədəbiyyatçısısan.\n"
        f"Sənin işin azərbaycanca `{genre}` janrında ədəbi mətnlər hazırlamaqdır. "
        "Mətnlər minimum 50 sözdən ibarət olmalı və yüksək ədəbi keyfiyyətdə yazılmalıdır."
        "Yaratdığın cümlələri maksimum fərqli sözlərdən istifadə edərək yarat."
        "Cavabınız mütləq düzgün JSON formatında olmalıdır. "
        "Bütün mətnlərdə dırnaq işarələri düzgün qapadılmalıdır."
    )
    
    for attempt in range(max_retries):
        try:
            # Run the API call in a thread to not block the event loop
            response = await asyncio.to_thread(
                together.chat.completions.create,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": transcript,
                    },
                ],
                model="meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo",
                temperature=temperature,
                response_format={
                    "type": "json_object",
                    "schema": LiteratureText.model_json_schema(),
                },
            )
            
            output = json.loads(response.choices[0].message.content)
            output["genre"] = genre
            
            return output
        
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"Request failed (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                print(f"Request failed after {max_retries} attempts: {e}")
                return {"text": f"Error generating text: {str(e)}", "genre": genre}

async def generate_samples(total_samples=100, concurrent_requests=50, output_file="data.json"):
    """Generate literature samples with controlled concurrency and append to existing file"""
    
    # Create a semaphore to limit concurrent requests
    semaphore = asyncio.Semaphore(concurrent_requests)
    
    # Load existing data if file exists
    existing_data = []
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            with open(output_file, "r", encoding="utf-8") as f:
                existing_data = json.load(f)
            print(f"Loaded {len(existing_data)} existing samples from {output_file}")
        except json.JSONDecodeError:
            print(f"Error reading {output_file}. Starting with empty dataset.")
    
    async def limited_generate():
        async with semaphore:
            return await generate_literature_async()
    
    # Create all tasks with semaphore control
    tasks = [limited_generate() for _ in range(total_samples)]
    
    # Execute tasks with progress bar
    new_results = await tqdm_asyncio.gather(*tasks, desc="Generating samples")
    
    # Combine existing data with new results
    combined_results = existing_data + new_results
    
    # Write results to file
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(combined_results, f, indent=4, ensure_ascii=False)
    
    print(f"Data generation complete. Added {len(new_results)} new samples for a total of {len(combined_results)} in {output_file}")
    return new_results

if __name__ == "__main__":
    start_time = time.time()
    
    # Run the async function
    asyncio.run(generate_samples(total_samples=100, concurrent_requests=20))
    
    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")