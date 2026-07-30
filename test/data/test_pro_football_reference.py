from bs4 import BeautifulSoup

from src.data.pro_football_reference import ProFootballReferenceScraper


class TestProFootballReferenceScraper():
    @classmethod
    def setup_class(cls):
        cls.scraper = ProFootballReferenceScraper(data_dir="test_data")

    def test_scrape_html_table__table_in_main_html(self):
        html = '''
        <html>
            <table id="fantasy">
                <thead><tr><th>Player</th><th>Points</th></tr></thead>
                <tbody><tr><td>John Doe</td><td>100</td></tr></tbody>
            </table>
        </html>
        '''
        html_page_soup = BeautifulSoup(html, 'lxml')
        df = self.scraper.scrape_html_table(html_page_soup, "fantasy", 2023)
        assert list(df.columns) == ["Player", "Points"]
        assert df.iloc[0]["Player"] == "John Doe"

    def test_scrape_html_table__table_in_html_comment(self):
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
        html_page_soup = BeautifulSoup(html, 'lxml')
        df = self.scraper.scrape_html_table(html_page_soup, "fantasy", 2023)
        assert list(df.columns) == ["Player", "Points"]
        assert df.iloc[0]["Player"] == "Jane Smith"
        assert df.iloc[0]["Points"] == "120"

    def test_scrape_html_table__table_not_found(self):
        html_page_soup = BeautifulSoup('<html></html>', 'lxml')
        df = self.scraper.scrape_html_table(html_page_soup, "fantasy", 2023)
        assert df.empty

    def test_scrape_html_table__more_header_cols_than_data_cols(self):
        html = '''
        <html>
            <table id="fantasy">
                <thead><tr><th>Extra</th><th>Player</th><th>Points</th></tr></thead>
                <tbody><tr><td>John Doe</td><td>100</td></tr></tbody>
            </table>
        </html>
        '''
        html_page_soup = BeautifulSoup(html, 'lxml')
        df = self.scraper.scrape_html_table(html_page_soup, "fantasy", 2023)

        assert list(df.columns) == ["Player", "Points"]
        assert df.iloc[0]["Points"] == "100"

    def test_scrape_html_table__fewer_header_cols_than_than_data_cols(self):
        html = '''
        <html>
            <table id="fantasy">
                <thead><tr><th>Player</th></tr></thead>
                <tbody><tr><td>John Doe</td><td>100</td></tr></tbody>
            </table>
        </html>
        '''
        html_page_soup = BeautifulSoup(html, 'lxml')
        df = self.scraper.scrape_html_table(html_page_soup, "fantasy", 2023)

        assert list(df.columns) == ["Player", "Unknown_Col_1"]
        assert df.iloc[0]["Player"] == "John Doe"
        assert df.iloc[0]["Unknown_Col_1"] == "100"
