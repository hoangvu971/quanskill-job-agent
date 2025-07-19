import argparse
from modules.resume_parser import ResumeParser
from modules.job_scraper import JobScraper
from modules.vector_store import Vectorizer
from modules.matcher import JobMatcher
from modules.qa_agent import LocalLLM


def main(resume_path: str, query: str):
    print(f" Parsing resume: {resume_path}")
    resume_parser = ResumeParser(resume_path)
    resume_text = resume_parser.extract_text()

    print(f" Scraping jobs for: {query}")
    scraper = JobScraper(search_term=query)
    df_jobs = scraper.scrape()

    print(f" Generating embeddings...")
    vectorizer = Vectorizer()
    resume_emb = vectorizer.embed_text(resume_text)
    job_embs = vectorizer.embed_texts(df_jobs["description"].tolist())

    print(f" Matching jobs...")
    matcher = JobMatcher()
    ranked_jobs = matcher.rank_jobs(resume_emb, job_embs, df_jobs, top_k=5)
    print(ranked_jobs[["title", "company", "similarity"]])

    # Optional Q&A
    llm = LocalLLM()
    job_desc = ranked_jobs.iloc[0]["description"]
    question = input("\nAsk a question about the top job: ")
    answer = llm.answer_question(job_desc, question)
    print(f"\n Answer: {answer}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Job Agent CLI")
    parser.add_argument("--resume", required=True, help="Path to resume PDF")
    parser.add_argument("--query", required=True, help="Job search query")
    args = parser.parse_args()

    main(args.resume, args.query)
