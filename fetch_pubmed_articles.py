from datetime import datetime, timedelta
import os
from Bio import Entrez
import time
import pandas as pd

# Use your email address to comply with PubMed's API requirements
Entrez.email = "cheikh-omar.ba@dna-medlabs.com"

def fetch_pubmed_articles_by_week(search_term, start_date, end_date, batch_size=1000, output_directory="./"):
    print(f"Starting to fetch articles for search term: {search_term}")
    
    all_articles = []
    total_articles_count = 0

    # Create output directory if not exists
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
        print(f"Created directory: {output_directory}")

    current_start_date = start_date

    while current_start_date < end_date:
        current_end_date = current_start_date + timedelta(weeks=1) - timedelta(days=1)
        if current_end_date > end_date:
            current_end_date = end_date

        start = 0
        total_count = 1  # Initialize total_count to enter the loop

        while start < total_count:
            try:
                # Create PubMed query for the given period
                query = f"{search_term} AND ({current_start_date.strftime('%Y/%m/%d')}:{current_end_date.strftime('%Y/%m/%d')})[pdat]"
                print(f"Fetching articles for {current_start_date.strftime('%Y/%m/%d')} to {current_end_date.strftime('%Y/%m/%d')}: Start {start} to {start + batch_size}")

                # Query PubMed to get article IDs
                handle = Entrez.esearch(db="pubmed", term=query, retmax=batch_size, retstart=start)
                record = Entrez.read(handle)
                total_count = int(record["Count"])
                total_articles_count += len(record["IdList"])
                id_list = record["IdList"]
                handle.close()

                if not id_list:
                    print(f"No articles found for the week {current_start_date.strftime('%Y/%m/%d')} to {current_end_date.strftime('%Y/%m/%d')}")
                    break

                # Fetch article details using the PubMed IDs
                ids = ",".join(id_list)
                fetch_handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml", retmode="text")
                articles_data = Entrez.read(fetch_handle)
                fetch_handle.close()

                print(f"Fetched {len(articles_data['PubmedArticle'])} articles for this period.")

                for article in articles_data['PubmedArticle']:
                    # Safely extract data with checks
                    try:
                        # First try to get 'PubMed_ID', if not present, fall back to 'PMID'
                        pubmed_id = article['MedlineCitation'].get('PubMed_ID', article['MedlineCitation'].get('PMID', None))
                        if not pubmed_id:
                            print(f"Missing expected key in article: 'PubMed_ID' or 'PMID' for article {article}")
                            continue
                    except KeyError:
                        print(f"Error accessing PubMed_ID or PMID in article: {article}")
                        continue

                    title = article['MedlineCitation']['Article']['ArticleTitle']
                    abstract = article['MedlineCitation']['Article'].get('Abstract', {}).get('AbstractText', ['No abstract available'])[0]
                    content = f"{title} {abstract}"

                    # Extract the full date (year/month/day) from 'PubDate'
                    pub_date = article['MedlineCitation']['Article']['Journal']['JournalIssue']['PubDate']
                    year = pub_date.get('Year', 'Unknown')
                    month = pub_date.get('Month', 'Unknown')
                    day = pub_date.get('Day', 'Unknown')

                    # Format date as YYYY/MM/DD (using 'Unknown' for missing parts)
                    publication_date = f"{year}/{month}/{day}" if year != 'Unknown' and month != 'Unknown' and day != 'Unknown' else 'Unknown'
                    journal = article['MedlineCitation']['Article']['Journal']['Title']

                    # Additional details
                    authors = article['MedlineCitation']['Article'].get('AuthorList', [])
                    authors_names = [f"{author['LastName']}, {author['ForeName']}" for author in authors if 'LastName' in author and 'ForeName' in author]
                    authors_str = ', '.join(authors_names) if authors_names else 'Unknown'

                    doi = article['MedlineCitation']['Article'].get('ArticleIdList', [])
                    doi_value = next((doi_item for doi_item in doi if doi_item.attributes['Source'] == 'doi'), 'Unknown')

                    article_type = article['MedlineCitation']['Article'].get('ArticleTypeList', ['Unknown'])[0]

                    language = article['MedlineCitation']['Article'].get('Language', 'Unknown')

                    mesh_terms = article['MedlineCitation']['MeshHeadingList'] if 'MeshHeadingList' in article['MedlineCitation'] else []
                    mesh_terms_str = ', '.join([mesh['DescriptorName'] for mesh in mesh_terms])

                    # Handle missing 'GrantList' gracefully
                    try:
                        grant_support = article['MedlineCitation']['Article'].get('GrantList', [])
                        grant_support_str = ', '.join([grant['GrantID'] for grant in grant_support]) if grant_support else 'None'
                    except KeyError:
                        grant_support_str = 'None'

                    # Add the extracted data to the list
                    all_articles.append([ 
                        pubmed_id, title, abstract, content, journal, publication_date, 
                        authors_str, doi_value, article_type, language, mesh_terms_str, grant_support_str
                    ])

                start += batch_size
                time.sleep(1)  # Respect PubMed API limits

            except Exception as e:
                print(f"Error fetching articles: {e}")
                break

        if all_articles:
            try:
                # Create DataFrame with columns in uppercase
                df = pd.DataFrame(all_articles, columns=[ 
                    "PUBMED_ID", "TITLE", "ABSTRACT", "CONTENT", "JOURNAL", "PUBLICATION DATE", 
                    "AUTHORS", "DOI", "ARTICLE TYPE", "LANGUAGE", "MESH TERMS", "GRANT SUPPORT"
                ])
                
                # Save the articles in a CSV file with a name based on the dates
                week_file = f"{output_directory}/pubmed_plants_articles_{current_start_date.strftime('%Y-%m-%d')}_to_{current_end_date.strftime('%Y-%m-%d')}.csv"
                df.to_csv(week_file, index=False)
                print(f"Articles saved to {week_file}")
            except Exception as save_error:
                print(f"Error saving file: {save_error}")

        all_articles = []  # Reset for the next period
        current_start_date += timedelta(weeks=1)

    print(f"Total articles fetched: {total_articles_count}")
