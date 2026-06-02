import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')


class TechStackRecommender:
    def __init__(self, dataset_path=None):
        if dataset_path:
            self.df = pd.read_csv(dataset_path)
        else:
            # Create sample dataset if no file provided
            self._create_sample_dataset()
        
        self.tfidf_matrix = None
        self.tfidf_vectorizer = None
        self.job_roles = None
        self._prepare_tfidf_matrix()
    
    def _create_sample_dataset(self):
        sample_data = {
            'job_role': [
                'Cloud Architect',
                'DevOps Engineer',
                'Data Scientist',
                'Machine Learning Engineer',
                'Backend Developer',
                'Frontend Developer',
                'Full Stack Developer',
                'Security Analyst',
                'AI Research Scientist',
                'Database Administrator',
                'Mobile App Developer',
                'Site Reliability Engineer'
            ],
            'skills': [
                'Python Cloud Computing AWS Automation Docker Kubernetes',
                'Python Linux Docker Jenkins CI/CD Automation Cloud',
                'Python Statistics Machine Learning SQL Data Visualization Pandas',
                'Python TensorFlow PyTorch Deep Learning MLOps Cloud Computing',
                'Python Django REST APIs SQL Git Algorithms',
                'JavaScript React CSS HTML TypeScript Web Design',
                'Python JavaScript React Django Node.js MongoDB SQL',
                'Network Security Cybersecurity Risk Analysis Python Scripting',
                'Python Mathematics Research Deep Learning NLP Computer Vision',
                'SQL PostgreSQL Database Tuning Backup Recovery Performance',
                'Kotlin Swift Android iOS React Native Mobile Development',
                'Python Go Kubernetes Monitoring Linux Automation SRE'
            ],
            'domain': [
                'Cloud',
                'DevOps',
                'Data Science',
                'AI/ML',
                'Backend',
                'Frontend',
                'Full Stack',
                'Security',
                'AI/ML',
                'Database',
                'Mobile',
                'DevOps'
            ],
            'avg_salary': [
                150000, 140000, 135000, 155000, 120000,
                115000, 130000, 125000, 160000, 110000,
                118000, 145000
            ]
        }
        self.df = pd.DataFrame(sample_data)
        print(f"[INFO] Created sample dataset with {len(self.df)} job roles.")
    
    def _prepare_tfidf_matrix(self):
        """
        Step 1: Feature Extraction using TF-IDF
        Converts skills text into weighted numerical vectors
        """
        print("[INFO] Preparing TF-IDF matrix...")
        
        # Initialize TF-IDF Vectorizer
        # - stop_words='english' removes common words
        # - lowercase=True standardizes text
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words='english',
            lowercase=True,
            min_df=1,  # Keep all terms
            max_df=1.0,
            ngram_range=(1, 2)  # Use both unigrams and bigrams for better matching
        )
        
        # Fit and transform the skills column
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.df['skills'])
        self.job_roles = self.df['job_role'].tolist()
        
        print(f"[INFO] TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        print(f"[INFO] Vocabulary size: {len(self.tfidf_vectorizer.get_feature_names_out())} words")
    
    def get_user_profile_vector(self, user_skills):
        # Join skills into a single string (same format as dataset)
        user_skills_text = ' '.join(user_skills)
        
        # Transform using the pre-trained vectorizer
        user_vector = self.tfidf_vectorizer.transform([user_skills_text])
        
        return user_vector
    
    def calculate_cosine_similarities(self, user_vector):
        # Compute cosine similarity between user and all job roles
        similarities = cosine_similarity(user_vector, self.tfidf_matrix)
        
        # Flatten to 1D array
        similarity_scores = similarities.flatten()
        
        # Create results DataFrame
        results_df = self.df.copy()
        results_df['similarity_score'] = similarity_scores
        
        return results_df
    
    def get_top_n_recommendations(self, user_skills, n=3, min_score_threshold=0.0):
        # Validate input
        if len(user_skills) < 3:
            print(f"[WARNING] Only {len(user_skills)} skills provided. Minimum 3 recommended for good results.")
        
        # Transform user input to vector
        user_vector = self.get_user_profile_vector(user_skills)
        
        # Calculate similarity scores
        results_df = self.calculate_cosine_similarities(user_vector)
        
        # Sort by similarity score descending
        results_df = results_df.sort_values('similarity_score', ascending=False)
        
        # Apply threshold filter
        results_df = results_df[results_df['similarity_score'] >= min_score_threshold]
        
        # Return Top N
        top_n = results_df.head(n)
        
        return top_n
    
    def display_recommendations(self, recommendations):
        if recommendations.empty:
            print("\n[ERROR] No recommendations found. Try different skills or lower the threshold.")
            return
        
        print("\n" + "="*70)
        print("🎯 TOP RECOMMENDED JOB ROLES".center(70))
        print("="*70)
        
        for idx, (_, row) in enumerate(recommendations.iterrows(), 1):
            similarity_pct = row['similarity_score'] * 100
            print(f"\n{idx}. {row['job_role']}")
            print(f"   📊 Match Score: {similarity_pct:.1f}%")
            print(f"   📂 Domain: {row['domain']}")
            print(f"   💰 Avg Salary: ${row['avg_salary']:,}")
            print(f"   🔧 Key Skills: {row['skills'][:100]}...")
        
        print("\n" + "="*70)
    
    def handle_cold_start(self, user_skills):
        if not user_skills or len(user_skills) == 0:
            print("[INFO] Cold Start Detected: No user skills provided.")
            print("[INFO] Using Trending Fallback - Showing globally popular job roles.")
            
            # Return top job roles by salary as "trending" fallback
            trending = self.df.sort_values('avg_salary', ascending=False).head(3)
            trending['similarity_score'] = 0.5  # Default medium score
            
            return trending
        
        return None


def run_recommendation_session():
    print("\n" + "🚀"*10)
    print(" DECODELABS - TECH STACK RECOMMENDER ".center(70, "🚀"))
    print("🚀"*10)
    print("\n📌 Content-Based Filtering using TF-IDF + Cosine Similarity")
    
    # Initialize recommender
    recommender = TechStackRecommender()
    
    while True:
        print("\n" + "-"*50)
        print("MAIN MENU".center(50))
        print("-"*50)
        print("1. Get Job Role Recommendations")
        print("2. View All Available Job Roles")
        print("3. About this Recommendation Engine")
        print("4. Exit")
        
        choice = input("\nEnter your choice (1-4): ").strip()
        
        if choice == '1':
            # Get user input
            print("\n🔧 ENTER YOUR SKILLS (minimum 3 recommended)")
            print("   Example: Python, Cloud Computing, Automation, SQL, JavaScript\n")
            
            skills_input = input("Your skills (comma-separated): ").strip()
            user_skills = [skill.strip().title() for skill in skills_input.split(',')]
            
            # Remove empty strings
            user_skills = [s for s in user_skills if s]
            
            if len(user_skills) == 0:
                # Handle cold start
                recommendations = recommender.handle_cold_start(user_skills)
                recommender.display_recommendations(recommendations)
            else:
                print(f"\n[INFO] Analyzing {len(user_skills)} skills: {', '.join(user_skills)}")
                print("[INFO] Calculating TF-IDF vectors and Cosine Similarity scores...")
                
                # Get recommendations
                recommendations = recommender.get_top_n_recommendations(user_skills, n=3)
                recommender.display_recommendations(recommendations)
                
                # Show similarity explanation
                print("\n📐 SIMILARITY EXPLANATION:")
                print("   • Cosine Similarity measures the ANGLE between user and job role vectors.")
                print("   • Scores range from 0 (no match) to 1 (perfect match).")
                print("   • TF-IDF weighting gives more importance to unique/rare skills.")
        
        elif choice == '2':
            print("\n📋 ALL AVAILABLE JOB ROLES:")
            print("-"*50)
            for idx, row in recommender.df.iterrows():
                print(f"{idx+1:2}. {row['job_role']:25} | {row['domain']:12} | Skills: {row['skills'][:50]}...")
        
        elif choice == '3':
            print("\n" + "="*60)
            print("📖 ABOUT THIS RECOMMENDATION ENGINE".center(60))
            print("="*60)
            print("""
┌─────────────────────────────────────────────────────────────────┐
│  ARCHITECTURE                                                   │
│  • Content-Based Filtering (not collaborative)                 │
│  • Input → Process → Output (IPO Model)                        │
│                                                                 │
│  MATHEMATICAL FOUNDATION                                        │
│  • TF-IDF: Term Frequency - Inverse Document Frequency         │
│    → Penalizes common words (Python, SQL)                       │
│    → Rewards rare, descriptive skills                          │
│  • Cosine Similarity (not Euclidean Distance)                  │
│    → Measures angle between vectors                            │
│    → Invariant to vector magnitude                             │
│                                                                 │
│  4-STEP PIPELINE                                                │
│  1. Ingestion: Capture 3+ user skills                          │
│  2. Scoring: Cosine similarity calculation                     │
│  3. Sorting: Descending order by score                         │
│  4. Filtering: Return Top-3 recommendations                    │
│                                                                 │
│  COLD START HANDLING                                            │
│  • Trending Fallback: Show high-salary roles when no input     │
└─────────────────────────────────────────────────────────────────┘
""")
        
        elif choice == '4':
            print("\n👋 Thank you for using Tech Stack Recommender!")
            print("   Mastered: Content-Based Filtering | TF-IDF | Cosine Similarity\n")
            break
        
        else:
            print("[ERROR] Invalid choice. Please enter 1-4.")


# Alternative: Direct function call for automated testing
def quick_recommend(skills_list):
    recommender = TechStackRecommender()
    return recommender.get_top_n_recommendations(skills_list, n=3)
# ========== MAIN EXECUTION ==========
if __name__ == "__main__":
    run_recommendation_session()