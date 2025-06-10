import unittest
from unittest.mock import patch

from src.scraper import ProFootballReferenceScraper

from bs4 import BeautifulSoup

class TestProFootballReferenceScraper(unittest.TestCase):
    def setUp(self):
        self.scraper = ProFootballReferenceScraper(data_dir="test_data")

    @patch.object(ProFootballReferenceScraper, '_get_soup')
    def test_table_in_main_html(self, mock_get_soup):
        html = '''
        <html>
            <table id="fantasy">
                <thead><tr><th>Player</th><th>Points</th></tr></thead>
                <tbody><tr><td>John Doe</td><td>100</td></tr></tbody>
            </table>
        </html>
        '''
        mock_get_soup.return_value = BeautifulSoup(html, 'lxml')
        df = self.scraper.scrape_html_table("fake_url", "fake_file", "fantasy", 2023)
        self.assertEqual(list(df.columns), ["Player", "Points"])
        self.assertEqual(df.iloc[0]["Player"], "John Doe")

    @patch.object(ProFootballReferenceScraper, '_get_soup')
    def test_table_in_html_comment(self, mock_get_soup):
        html = '''
        <html>
            <!--
            <table id="fantasy">
                <thead><tr><th>Player</th><th>Points</th></tr></thead>
                <tbody><tr><td>Jane Smith</td><td>120</td></tr></tbody>
            </table>
            -->
        </html>
        '''
        mock_get_soup.return_value = BeautifulSoup(html, 'lxml')
        df = self.scraper.scrape_html_table("fake_url", "fake_file", "fantasy", 2023)
        self.assertEqual(list(df.columns), ["Player", "Points"])
        self.assertEqual(df.iloc[0]["Player"], "Jane Smith")
        self.assertEqual(df.iloc[0]["Points"], "120")

    @patch.object(ProFootballReferenceScraper, '_get_soup')
    def test_table_not_found(self, mock_get_soup):
        html = '<html></html>'
        mock_get_soup.return_value = BeautifulSoup(html, 'lxml')
        df = self.scraper.scrape_html_table("fake_url", "fake_file", "fantasy", 2023)
        self.assertTrue(df.empty)

    @patch.object(ProFootballReferenceScraper, '_get_soup')
    def test_header_more_than_body(self, mock_get_soup):
        html = '''
        <html>
            <table id="fantasy">
                <thead><tr><th>Extra</th><th>Player</th><th>Points</th></tr></thead>
                <tbody><tr><td>John Doe</td><td>100</td></tr></tbody>
            </table>
        </html>
        '''
        mock_get_soup.return_value = BeautifulSoup(html, 'lxml')
        df = self.scraper.scrape_html_table("fake_url", "fake_file", "fantasy", 2023)

        self.assertEqual(list(df.columns), ["Player", "Points"])
        self.assertEqual(df.iloc[0]["Points"], "100")

    @patch.object(ProFootballReferenceScraper, '_get_soup')
    def test_header_fewer_than_body(self, mock_get_soup):
        html = '''
        <html>
            <table id="fantasy">
                <thead><tr><th>Player</th></tr></thead>
                <tbody><tr><td>John Doe</td><td>100</td></tr></tbody>
            </table>
        </html>
        '''
        mock_get_soup.return_value = BeautifulSoup(html, 'lxml')
        df = self.scraper.scrape_html_table("fake_url", "fake_file", "fantasy", 2023)

        self.assertEqual(list(df.columns), ["Player", "Unknown_Col_1"])
        self.assertEqual(df.iloc[0]["Player"], "John Doe")
        self.assertEqual(df.iloc[0]["Unknown_Col_1"], "100")

if __name__ == '__main__':
    unittest.main() 
