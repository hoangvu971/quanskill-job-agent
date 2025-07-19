import jobspy
import pandas as pd
from jobspy import scrape_jobs


class JobScraper:
    """
    Scrapes job postings from supported platforms.
    """

    def __init__(
        self,
        search_term: str,
        location: str = "Remote",
        sites=None,
        country_indeed: str = "Vietnam",
        num_results: int = 20,
    ):
        self.search_term = search_term
        self.location = location
        self.sites = sites or ["linkedin", "indeed"]
        self.num_results = num_results
        self.country_indeed = country_indeed

    def scrape(self) -> pd.DataFrame:
        """
        Scrape job postings and return a pandas DataFrame.
        """
        results = scrape_jobs(
            site_name=self.sites,
            search_term=self.search_term,
            location=self.location,
            results_wanted=self.num_results,
            country_indeed=self.country_indeed,
            hours_old=72,
        )

        df = pd.DataFrame(results)

        # Keep only essential columns
        essential_cols = [
            "title",
            "company",
            "location",
            "description",
            "date_posted",
            "job_url",
        ]
        df = df[[col for col in essential_cols if col in df.columns]].dropna(
            subset=["description"]
        )

        return df.reset_index(drop=True)
