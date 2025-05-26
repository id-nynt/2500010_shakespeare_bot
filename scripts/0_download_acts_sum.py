import requests
from bs4 import BeautifulSoup
import json
import time
import re
from urllib.parse import urljoin, urlparse
import csv
import os

class ShakespeareScraper:
    def __init__(self, output_dir="data/raw/summary_act"):
        self.base_url = "https://www.shakespeare.org.uk"
        self.main_url = "https://www.shakespeare.org.uk/explore-shakespeare/shakespedia/shakespeares-plays/"
        self.output_dir = output_dir
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        self.plays_data = []
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"Output directory: {os.path.abspath(self.output_dir)}")
        
    def get_soup(self, url):
        """Get BeautifulSoup object for a given URL with error handling"""
        try:
            print(f"Fetching: {url}")
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except requests.RequestException as e:
            print(f"Error fetching {url}: {e}")
            return None
    
    def extract_play_links(self):
        """Extract all play links from the main page"""
        soup = self.get_soup(self.main_url)
        
        # Comprehensive list of all Shakespeare plays
        all_shakespeare_plays = [
            "alls-well-ends-well", "antony-and-cleopatra", "as-you-like-it",
                "comedy-of-errors", "coriolanus", "cymbeline", "hamlet",
                "henry-iv-part-1", "henry-iv-part-2", "henry-v", "henry-vi-part-1",
                "henry-vi-part-2", "henry-vi-part-3", "henry-viii", "julius-caesar", "king-john",
                "king-lear", "loves-labours-lost", "macbeth", "measure-measure",
                "merchant-venice", "merry-wives-windsor", "midsummer-nights-dream",
                "much-ado-about-nothing", "othello-moor-venice", "pericles-prince-tyre", "richard-ii", "richard-iii",
                "romeo-and-juliet", "taming-of-the-shrew", "tempest", "timon-athens",
                "titus-andronicus", "troilus-and-cressida", "twelfth-night",
                "two-gentlemen-verona", "winters-tale"
        ]
        
        play_links = []
        found_links = set()
        
        if soup:
            print("Trying to extract play links from main page...")
            
            # Method 1: Look for all links containing shakespeares-plays
            all_links = soup.find_all('a', href=True)
            print(f"Found {len(all_links)} total links on main page")
            
            for link in all_links:
                href = link['href']
                
                # Check if this is a play link
                if 'shakespeares-plays' in href and href.endswith('/'):
                    # Extract play slug
                    parts = href.strip('/').split('/')
                    if len(parts) >= 2:
                        play_slug = parts[-1]
                        
                        # Verify this is a valid Shakespeare play
                        if play_slug in all_shakespeare_plays:
                            full_url = urljoin(self.base_url, href)
                            title = link.get_text(strip=True) or play_slug.replace('-', ' ').title()
                            
                            if full_url not in found_links:
                                found_links.add(full_url)
                                play_links.append({
                                    'title': title,
                                    'url': full_url,
                                    'slug': play_slug
                                })
            
            print(f"Method 1 found {len(play_links)} play links from main page")
        
        # Method 2: Use comprehensive fallback list regardless
        # This ensures we get all plays even if the website structure changes
        if len(play_links) < 30:  # Expected around 37-39 plays
            print("Using comprehensive play list to ensure all plays are covered...")
            
            for play_slug in all_shakespeare_plays:
                play_url = f"{self.main_url}{play_slug}/"
                title = play_slug.replace('-', ' ').title()
                
                # Avoid duplicates
                if play_url not in found_links:
                    found_links.add(play_url)
                    play_links.append({
                        'title': title,
                        'url': play_url,
                        'slug': play_slug
                    })
        
        # Sort plays alphabetically for consistent processing
        play_links.sort(key=lambda x: x['slug'])
        
        print(f"\n=== Final Play List ({len(play_links)} plays) ===")
        for i, link in enumerate(play_links, 1):
            print(f"{i:2d}. {link['title']:35} | {link['slug']}")
        
        return play_links
    
    def extract_play_summary_and_acts(self, play_url, play_title, play_slug):
        """Extract play summary and act summaries from a play's page"""
        soup = self.get_soup(play_url)
        if not soup:
            return {}
        
        print(f"Scraping: {play_title}")
        
        extracted_data = {
            'general_summary': '',
            'acts': {}
        }
        
        # Extract the main summary (usually the first paragraph after the title)
        # Look for the pattern: "Play Name Summary" followed by summary text
        summary_patterns = [
            f"{play_title} Summary",
            "Summary",
            f"{play_title.split()[0]} Summary"  # First word + Summary
        ]
        
        # Find the general summary
        for pattern in summary_patterns:
            heading = soup.find(lambda tag: tag.name in ['h1', 'h2', 'h3'] and 
                              pattern.lower() in tag.get_text().lower())
            if heading:
                # Get the next paragraph or div after the heading
                next_element = heading.find_next_sibling(['p', 'div'])
                if next_element:
                    summary_text = next_element.get_text(strip=True)
                    if len(summary_text) > 50:  # Substantial summary
                        extracted_data['general_summary'] = summary_text
                        break
        
        # If no summary found with heading, look for the first substantial paragraph
        if not extracted_data['general_summary']:
            paragraphs = soup.find_all('p')
            for p in paragraphs:
                text = p.get_text(strip=True)
                # Look for summary-like content (mentions character names, plot elements)
                if (len(text) > 100 and 
                    not text.startswith('Act ') and
                    any(word in text.lower() for word in ['king', 'duke', 'lord', 'lady', 'prince', 'marry', 'love', 'death', 'war'])):
                    extracted_data['general_summary'] = text
                    break
        
        # Extract Act summaries
        # Look for "Act I", "Act II", etc. followed by content
        act_pattern = re.compile(r'^Act\s+([IVX]+|[0-9]+)\s*$', re.IGNORECASE)
        
        # Find all headings that match act pattern
        headings = soup.find_all(['h2', 'h3', 'h4', 'strong', 'b'])
        
        for heading in headings:
            heading_text = heading.get_text(strip=True)
            act_match = act_pattern.match(heading_text)
            
            if act_match:
                act_number = act_match.group(1)
                print(f"  Found: Act {act_number}")
                
                # Collect all content until the next act or end
                content_parts = []
                current = heading
                
                # Look for content after this heading
                while current:
                    current = current.find_next_sibling()
                    if not current:
                        break
                    
                    # Stop if we hit another act heading
                    if (current.name in ['h2', 'h3', 'h4', 'strong', 'b'] and 
                        act_pattern.match(current.get_text(strip=True))):
                        break
                    
                    # Collect text content
                    if current.name in ['p', 'div']:
                        text = current.get_text(strip=True)
                        if text and len(text) > 20:  # Substantial content
                            content_parts.append(text)
                
                if content_parts:
                    act_summary = ' '.join(content_parts)
                    extracted_data['acts'][f"act_{act_number.lower()}"] = act_summary
        
        # Alternative approach: look for paragraphs that start with act mentions
        if not extracted_data['acts']:
            paragraphs = soup.find_all('p')
            current_act = None
            
            for p in paragraphs:
                text = p.get_text(strip=True)
                
                # Check if this paragraph starts a new act description
                act_start_match = re.match(r'^Act\s+([IVX]+|[0-9]+)', text, re.IGNORECASE)
                if act_start_match:
                    current_act = act_start_match.group(1).lower()
                    extracted_data['acts'][f"act_{current_act}"] = text
                    print(f"  Found Act content: Act {current_act}")
                elif current_act and len(text) > 50:
                    # Continue adding to current act if it's substantial content
                    # and doesn't start a new act
                    if not re.match(r'^Act\s+([IVX]+|[0-9]+)', text, re.IGNORECASE):
                        if f"act_{current_act}" in extracted_data['acts']:
                            extracted_data['acts'][f"act_{current_act}"] += ' ' + text
        
        # Extract quotes if present
        quotes = []
        quote_elements = soup.find_all(['blockquote', 'em', 'i'])
        for element in quote_elements:
            quote_text = element.get_text(strip=True)
            if len(quote_text) > 20 and len(quote_text) < 200:  # Reasonable quote length
                quotes.append(quote_text)
        
        if quotes:
            extracted_data['quotes'] = quotes[:3]  # Keep top 3 quotes
        
        return extracted_data
    
    def scrape_all_plays(self, limit=None):
        """Main method to scrape all plays"""
        print("Starting to scrape Shakespeare play summaries...")
        
        # Get all play links
        play_links = self.extract_play_links()
        
        if not play_links:
            print("No play links found. Please check the website structure.")
            return []
        
        # Limit number of plays if specified (useful for testing)
        if limit:
            play_links = play_links[:limit]
            print(f"Limited to first {limit} plays for testing")
        
        # Scrape each play
        success_count = 0
        for i, play_info in enumerate(play_links):
            try:
                print(f"\n[{i+1}/{len(play_links)}] Processing: {play_info['title']}")
                
                play_data = self.extract_play_summary_and_acts(
                    play_info['url'], 
                    play_info['title'],
                    play_info['slug']
                )
                
                if play_data['general_summary'] or play_data['acts']:
                    self.plays_data.append({
                        'play_title': play_info['title'],
                        'play_slug': play_info['slug'],
                        'play_url': play_info['url'],
                        'general_summary': play_data['general_summary'],
                        'acts': play_data['acts'],
                        'quotes': play_data.get('quotes', []),
                        'total_acts': len(play_data['acts'])
                    })
                    success_count += 1
                    print(f"  ✓ Summary: {'Yes' if play_data['general_summary'] else 'No'}")
                    print(f"  ✓ Acts found: {len(play_data['acts'])}")
                    if play_data.get('quotes'):
                        print(f"  ✓ Quotes found: {len(play_data['quotes'])}")
                else:
                    print(f"  ✗ No content found")
                
                # Save intermediate results every 10 plays in case of interruption
                if (i + 1) % 10 == 0:
                    print(f"\n--- Saving intermediate results ({success_count} successful so far) ---")
                    self.save_to_json(f"shakespeare_summaries_partial_{i+1}.json")
                
                # Be respectful to the server
                time.sleep(2)
                
            except Exception as e:
                print(f"  ✗ Error processing {play_info['title']}: {e}")
                continue
        
        print(f"\n=== Processing Complete ===")
        print(f"Successfully processed: {success_count}/{len(play_links)} plays")
        
        return self.plays_data
    
    def save_to_json(self, filename="shakespeare_summaries.json"):
        """Save scraped data to JSON file"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.plays_data, f, indent=2, ensure_ascii=False)
        print(f"Complete dataset saved to {filepath}")
    
    def save_to_csv(self, filename="shakespeare_summaries.csv"):
        """Save scraped data to CSV file for easier analysis"""
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Play Title', 'Content Type', 'Content', 'Word Count'])
            
            for play in self.plays_data:
                # General summary
                if play['general_summary']:
                    word_count = len(play['general_summary'].split())
                    writer.writerow([
                        play['play_title'],
                        'General Summary',
                        play['general_summary'],
                        word_count
                    ])
                
                # Act summaries
                for act_key, act_content in play['acts'].items():
                    word_count = len(act_content.split())
                    writer.writerow([
                        play['play_title'],
                        f'Act Summary - {act_key}',
                        act_content,
                        word_count
                    ])
                
                # Quotes
                for i, quote in enumerate(play.get('quotes', [])):
                    writer.writerow([
                        play['play_title'],
                        f'Quote {i+1}',
                        quote,
                        len(quote.split())
                    ])
        
        print(f"CSV data saved to {filepath}")
    
    def save_individual_play_files(self):
        """Save each play's summaries to individual JSON files"""
        for play in self.plays_data:
            # Create safe filename from play title
            safe_title = re.sub(r'[^\w\s-]', '', play['play_title'])
            safe_title = re.sub(r'[-\s]+', '_', safe_title).lower()
            filename = f"{safe_title}_summaries.json"
            filepath = os.path.join(self.output_dir, filename)
            
            play_data = {
                'title': play['play_title'],
                'slug': play['play_slug'],
                'url': play['play_url'],
                'general_summary': play['general_summary'],
                'acts': play['acts'],
                'quotes': play.get('quotes', [])
            }
            
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(play_data, f, indent=2, ensure_ascii=False)
        
        print(f"Individual play files saved to {self.output_dir}")
    
    def print_summary_stats(self):
        """Print statistics about scraped data"""
        if not self.plays_data:
            print("No data scraped.")
            return
        
        total_plays = len(self.plays_data)
        plays_with_summary = sum(1 for play in self.plays_data if play['general_summary'])
        total_acts = sum(play['total_acts'] for play in self.plays_data)
        total_quotes = sum(len(play.get('quotes', [])) for play in self.plays_data)
        
        print(f"\n=== Scraping Results ===")
        print(f"Total plays processed: {total_plays}")
        print(f"Plays with general summary: {plays_with_summary}")
        print(f"Total act summaries found: {total_acts}")
        print(f"Total quotes found: {total_quotes}")
        print(f"Average acts per play: {total_acts/total_plays:.1f}")
        
        print(f"\nDetailed breakdown:")
        for play in self.plays_data:
            summary_status = "✓" if play['general_summary'] else "✗"
            quote_count = len(play.get('quotes', []))
            print(f"  {play['play_title']:30} | Summary: {summary_status} | Acts: {play['total_acts']:2} | Quotes: {quote_count}")

def main():
    """Main execution function"""
    scraper = ShakespeareScraper()
    
    try:
        # Scrape all plays (you can add limit=10 to test with fewer plays first)
        plays_data = scraper.scrape_all_plays(limit=37)  # Start with 5 plays for testing
        
        if plays_data:
            # Save results
            scraper.save_to_json()
            scraper.save_to_csv()
            scraper.save_individual_play_files()
            scraper.print_summary_stats()
            
            print(f"\n=== Files saved in: {scraper.output_dir} ===")
            print("- shakespeare_summaries.json (complete dataset)")
            print("- shakespeare_summaries.csv (tabular format)")
            print("- individual_play_*.json (separate files for each play)")
        else:
            print("No data was scraped. Please check the website structure or network connection.")
    
    except KeyboardInterrupt:
        print("\nScraping interrupted by user.")
        if scraper.plays_data:
            print("Saving partial results...")
            scraper.save_to_json("partial_shakespeare_summaries.json")
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()