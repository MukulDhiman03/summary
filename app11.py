import json
import os
import pickle
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from datetime import datetime

# LangChain imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

@dataclass
class TopicAnalysis:
    topics: List[Dict[str, Any]]
    overarching_themes: List[str]
    narrative_flow: str
    importance_hierarchy: Dict[str, List[str]]

@dataclass
class ValidationResult:
    is_complete: bool
    suggestions: List[str]
    completeness_score: float

@dataclass
class CoherenceResult:
    needs_improvement: bool
    suggestions: List[str]
    coherence_score: float

class ChapterVectorStore:
    """
    Separate class for managing FAISS vector database operations
    Handles chapter embeddings, storage, and retrieval
    """
    
    def __init__(self, index_path: str, embedding_model: GoogleGenerativeAIEmbeddings):
        """
        Initialize the vector store with LangChain FAISS
        
        Args:
            index_path: Path to store FAISS index
            embedding_model: Google Generative AI embeddings model
        """
        self.index_path = index_path
        self.embedding_model = embedding_model
        self.metadata_path = f"{index_path}_metadata.pkl"
        self.vector_store = None
        self.chapter_metadata = {}
        
        self._load_or_create_vector_store()
    
    def _load_or_create_vector_store(self):
        """Load existing FAISS vector store or create new one"""
        try:
            if os.path.exists(self.index_path):
                # Load existing FAISS index using LangChain
                self.vector_store = FAISS.load_local(
                    self.index_path, 
                    self.embedding_model,
                    allow_dangerous_deserialization=True
                )
                print(f"Loaded existing FAISS index from {self.index_path}")
                
                # Load metadata
                if os.path.exists(self.metadata_path):
                    with open(self.metadata_path, 'rb') as f:
                        self.chapter_metadata = pickle.load(f)
                    print(f"Loaded metadata for {len(self.chapter_metadata)} chapters")
            else:
                # Create new empty vector store - will be created when first document is added
                self.vector_store = None
                self.chapter_metadata = {}
                print("Initialized new vector store")
                
        except Exception as e:
            print(f"Error loading vector store: {e}")
            self.vector_store = None
            self.chapter_metadata = {}
    
    def add_chapter(self, chapter_id: str, summary_text: str, full_result: Dict[str, Any]):
        """
        Add a chapter summary to the vector store
        
        Args:
            chapter_id: Unique identifier for the chapter
            summary_text: Chapter summary text to embed
            full_result: Complete chapter analysis results
        """
        try:
            # Create document with metadata
            doc = Document(
                page_content=summary_text,
                metadata={
                    "chapter_id": chapter_id,
                    "book_title": full_result.get("book_title", ""),
                    "chapter_title": full_result.get("chapter_title", ""),
                    "timestamp": full_result.get("timestamp", datetime.now().isoformat())
                }
            )
            
            if self.vector_store is None:
                # Create new vector store with first document
                self.vector_store = FAISS.from_documents([doc], self.embedding_model)
            else:
                # Add to existing vector store
                self.vector_store.add_documents([doc])
            
            # Store complete metadata separately
            self.chapter_metadata[chapter_id] = {
                "summary": summary_text,
                "full_result": full_result,
                "document_id": len(self.chapter_metadata)  # Simple ID tracking
            }
            
            # Save to disk
            self._save_vector_store()
            print(f"Added chapter {chapter_id} to vector store")
            
        except Exception as e:
            print(f"Error adding chapter to vector store: {e}")
    
    def search_similar_chapters(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chapters using semantic similarity
        
        Args:
            query: Search query text
            k: Number of similar chapters to return
            
        Returns:
            List of similar chapters with similarity scores
        """
        if self.vector_store is None:
            return []
        
        try:
            # Perform similarity search with scores
            results = self.vector_store.similarity_search_with_score(query, k=k)
            
            similar_chapters = []
            for doc, score in results:
                chapter_id = doc.metadata.get("chapter_id")
                if chapter_id in self.chapter_metadata:
                    result = {
                        "chapter_id": chapter_id,
                        "summary": doc.page_content,
                        "similarity_score": float(score),
                        "metadata": doc.metadata,
                        "full_result": self.chapter_metadata[chapter_id]["full_result"]
                    }
                    similar_chapters.append(result)
            
            return similar_chapters
            
        except Exception as e:
            print(f"Error searching similar chapters: {e}")
            return []
    
    def get_chapter_by_id(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chapter by its ID
        
        Args:
            chapter_id: Unique identifier for the chapter
            
        Returns:
            Chapter data if found, None otherwise
        """
        return self.chapter_metadata.get(chapter_id, {}).get("full_result")
    
    def get_all_chapters(self) -> List[str]:
        """Get list of all chapter IDs in the vector store"""
        return list(self.chapter_metadata.keys())
    
    def delete_chapter(self, chapter_id: str) -> bool:
        """
        Delete a chapter from the vector store
        Note: FAISS doesn't support individual document deletion easily,
        so this marks it as deleted in metadata
        """
        if chapter_id in self.chapter_metadata:
            self.chapter_metadata[chapter_id]["deleted"] = True
            self._save_vector_store()
            return True
        return False
    
    def _save_vector_store(self):
        """Save vector store and metadata to disk"""
        try:
            if self.vector_store is not None:
                # Save FAISS index
                self.vector_store.save_local(self.index_path)
                
                # Save metadata
                with open(self.metadata_path, 'wb') as f:
                    pickle.dump(self.chapter_metadata, f)
                
                print(f"Saved vector store to {self.index_path}")
        except Exception as e:
            print(f"Error saving vector store: {e}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics about the vector store"""
        active_chapters = sum(1 for meta in self.chapter_metadata.values() 
                            if not meta.get("deleted", False))
        
        return {
            "total_chapters": len(self.chapter_metadata),
            "active_chapters": active_chapters,
            "deleted_chapters": len(self.chapter_metadata) - active_chapters,
            "vector_store_size": len(self.vector_store.docstore._dict) if self.vector_store else 0
        }

class ChapterSummarizationPipeline:
    """
    Main pipeline for chapter summarization with Google Gemini models
    """
    
    def __init__(self, 
                 index_path: str,
                 emb_model_name: str = "models/embedding-001",
                 llm_model_name: str = "gemini-2.0-flash"):
        """
        Initialize the pipeline with Google Gemini models
        
        Args:
            index_path: Path for FAISS vector store
            emb_model_name: Google embedding model name
            llm_model_name: Google LLM model name
        """
        self.index_path = index_path
        
        # Initialize Google models
        self.embedding_model = GoogleGenerativeAIEmbeddings(model=emb_model_name)
        self.llm = ChatGoogleGenerativeAI(
            model=llm_model_name, 
            temperature=0.1, 
            max_output_tokens=3024
        )
        
        # Initialize vector store
        self.vector_store = ChapterVectorStore(index_path, self.embedding_model)
        
        print(f"Initialized pipeline with embedding model: {emb_model_name}")
        print(f"Initialized pipeline with LLM model: {llm_model_name}")
        print(f"Vector store path: {index_path}")
    
    def extract_topics_and_context(self, chapter_text: str) -> TopicAnalysis:
        """
        Stage 1: Analyze chapter and extract comprehensive topic information
        """
        prompt = f"""
        Analyze this chapter and provide a comprehensive topic analysis:

        CHAPTER TEXT:
        {chapter_text}

        Please provide your analysis in the following JSON format:
        {{
            "topics": [
                {{
                    "name": "Topic Name",
                    "description": "Brief description",
                    "key_quotes": ["quote1", "quote2"],
                    "importance_level": "Primary|Secondary|Supporting",
                    "relationships": ["related_topic1", "related_topic2"],
                    "merge_potential": "Can be merged with X" or "Standalone"
                }}
            ],
            "overarching_themes": ["theme1", "theme2"],
            "narrative_flow": "Description of how the chapter progresses",
            "importance_hierarchy": {{
                "Primary": ["topic1", "topic2"],
                "Secondary": ["topic3", "topic4"],
                "Supporting": ["topic5"]
            }}
        }}
        
        Focus on:
        - Identifying ALL important topics (don't miss subtle ones)
        - Capturing relationships between topics
        - Noting the chapter's overall narrative structure
        - Providing rich context for each topic
        
        Ensure the response is valid JSON format.
        """
        
        try:
            response = self.llm.invoke(prompt)
            response_text = response.content
            
            # Try to parse JSON response
            analysis_data = json.loads(response_text)
            return TopicAnalysis(
                topics=analysis_data["topics"],
                overarching_themes=analysis_data["overarching_themes"],
                narrative_flow=analysis_data["narrative_flow"],
                importance_hierarchy=analysis_data["importance_hierarchy"]
            )
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
            return self._parse_fallback_analysis(response_text)
        except Exception as e:
            print(f"Error in topic extraction: {e}")
            return self._create_default_analysis()
    
    def validate_completeness(self, topic_analysis: TopicAnalysis, chapter_text: str) -> ValidationResult:
        """
        Quality gate: Check if all important topics were captured
        """
        prompt = f"""
        Review this topic analysis for completeness:
        
        ORIGINAL CHAPTER: {chapter_text[:2000]}... [truncated for analysis]
        
        IDENTIFIED TOPICS: {[topic['name'] for topic in topic_analysis.topics]}
        
        Evaluate:
        1. Are there any missing important topics?
        2. Are the relationships between topics well captured?
        3. Does the analysis miss any key themes?
        
        Respond in JSON format:
        {{
            "is_complete": true/false,
            "completeness_score": 0.0-1.0,
            "suggestions": ["suggestion1", "suggestion2"]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            result_data = json.loads(response.content)
            return ValidationResult(
                is_complete=result_data["is_complete"],
                suggestions=result_data["suggestions"],
                completeness_score=result_data["completeness_score"]
            )
        except Exception as e:
            print(f"Error in validation: {e}")
            return ValidationResult(is_complete=True, suggestions=[], completeness_score=0.8)
    
    def generate_flexible_summaries(self, chapter_text: str, topic_analysis: TopicAnalysis) -> Dict[str, str]:
        """
        Stage 2: Create summaries with structural flexibility
        """
        prompt = f"""
        Create detailed summaries for each topic area from this chapter.
        
        CHAPTER: {chapter_text}
        
        TOPIC ANALYSIS: {json.dumps([topic for topic in topic_analysis.topics], indent=2)}
        
        OVERARCHING THEMES: {topic_analysis.overarching_themes}
        
        NARRATIVE FLOW: {topic_analysis.narrative_flow}
        
        Instructions:
        - Feel free to merge related topics if they flow better together
        - Split complex topics if they need separate treatment
        - Maintain connections between topics as identified
        - Include brief transitional context between summaries
        - Preserve the chapter's overall narrative arc
        
        Provide summaries in JSON format:
        {{
            "topic_name_1": "Detailed summary with context and connections...",
            "topic_name_2": "Another summary...",
            "merged_topic_x_y": "Combined summary if topics were merged...",
            "transitions": {{
                "topic1_to_topic2": "Brief connecting context",
                "topic2_to_topic3": "Another transition"
            }}
        }}
        
        Ensure the response is valid JSON.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return json.loads(response.content)
        except Exception as e:
            print(f"Error in summary generation: {e}")
            # Fallback: create basic summaries
            return {topic['name']: topic['description'] for topic in topic_analysis.topics}
    
    def check_summary_coherence(self, topic_summaries: Dict[str, str]) -> CoherenceResult:
        """
        Quality gate: Ensure summaries flow logically together
        """
        prompt = f"""
        Evaluate the coherence and flow of these topic summaries:
        
        SUMMARIES: {json.dumps(topic_summaries, indent=2)}
        
        Check for:
        1. Logical flow between topics
        2. Consistent tone and style
        3. Adequate transitions
        4. Overall narrative coherence
        
        Respond in JSON:
        {{
            "needs_improvement": true/false,
            "coherence_score": 0.0-1.0,
            "suggestions": ["improvement1", "improvement2"]
        }}
        """
        
        try:
            response = self.llm.invoke(prompt)
            result_data = json.loads(response.content)
            return CoherenceResult(
                needs_improvement=result_data["needs_improvement"],
                suggestions=result_data["suggestions"],
                coherence_score=result_data["coherence_score"]
            )
        except Exception as e:
            print(f"Error in coherence check: {e}")
            return CoherenceResult(needs_improvement=False, suggestions=[], coherence_score=0.8)
    
    def synthesize_final_summary(self, chapter_text: str, topic_analysis: TopicAnalysis, 
                               topic_summaries: Dict[str, str]) -> str:
        """
        Stage 3: Integrate summaries into cohesive chapter overview
        """
        prompt = f"""
        Create a final, cohesive chapter summary by integrating these topic summaries:
        
        ORIGINAL CHAPTER LENGTH: {len(chapter_text.split())} words
        
        OVERARCHING THEMES: {topic_analysis.overarching_themes}
        
        NARRATIVE FLOW: {topic_analysis.narrative_flow}
        
        TOPIC SUMMARIES: {json.dumps(topic_summaries, indent=2)}
        
        Create a flowing, comprehensive chapter summary that:
        - Begins with the chapter's main purpose/thrust
        - Integrates all topic summaries naturally
        - Maintains logical progression
        - Concludes with the chapter's significance
        - Reads as a unified piece, not separate sections
        
        Target length: 300-500 words for a comprehensive overview.
        Write in clear, engaging prose.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            print(f"Error in final synthesis: {e}")
            return "Summary generation failed. Please check the logs for details."
    
    def refine_topic_analysis(self, initial_analysis: TopicAnalysis, suggestions: List[str]) -> TopicAnalysis:
        """
        Iterative improvement of topic extraction based on validation feedback
        """
        prompt = f"""
        Refine this topic analysis based on the following suggestions:
        
        CURRENT ANALYSIS: {json.dumps({
            'topics': initial_analysis.topics,
            'themes': initial_analysis.overarching_themes,
            'flow': initial_analysis.narrative_flow
        }, indent=2)}
        
        IMPROVEMENT SUGGESTIONS: {suggestions}
        
        Provide the refined analysis in the same JSON format as the original analysis.
        Ensure valid JSON format.
        """
        
        try:
            response = self.llm.invoke(prompt)
            analysis_data = json.loads(response.content)
            return TopicAnalysis(
                topics=analysis_data["topics"],
                overarching_themes=analysis_data["overarching_themes"],
                narrative_flow=analysis_data["narrative_flow"],
                importance_hierarchy=analysis_data.get("importance_hierarchy", {})
            )
        except Exception as e:
            print(f"Error in refinement: {e}")
            return initial_analysis
    
    def improve_summary_flow(self, summaries: Dict[str, str], suggestions: List[str]) -> Dict[str, str]:
        """
        Enhance connections between topic summaries
        """
        prompt = f"""
        Improve the flow and coherence of these summaries:
        
        CURRENT SUMMARIES: {json.dumps(summaries, indent=2)}
        
        IMPROVEMENT SUGGESTIONS: {suggestions}
        
        Provide improved summaries in the same JSON format.
        Focus on better transitions and coherent narrative flow.
        """
        
        try:
            response = self.llm.invoke(prompt)
            return json.loads(response.content)
        except Exception as e:
            print(f"Error in flow improvement: {e}")
            return summaries
    
    def summarize_chapter(self, chapter_text: str, chapter_id: str, 
                         book_title: str = "", chapter_title: str = "") -> Dict[str, Any]:
        """
        Main pipeline orchestrator
        """
        print(f"Processing chapter: {chapter_id}")
        
        # Stage 1: Topic Extraction
        print("Stage 1: Extracting topics and context...")
        topic_analysis = self.extract_topics_and_context(chapter_text)
        
        # Quality Gate 1
        print("Quality Gate 1: Validating completeness...")
        validation = self.validate_completeness(topic_analysis, chapter_text)
        if not validation.is_complete:
            print("Refining topic analysis based on validation...")
            topic_analysis = self.refine_topic_analysis(topic_analysis, validation.suggestions)
        
        # Stage 2: Flexible Summarization
        print("Stage 2: Generating flexible summaries...")
        topic_summaries = self.generate_flexible_summaries(chapter_text, topic_analysis)
        
        # Quality Gate 2
        print("Quality Gate 2: Checking summary coherence...")
        coherence_check = self.check_summary_coherence(topic_summaries)
        if coherence_check.needs_improvement:
            print("Improving summary flow...")
            topic_summaries = self.improve_summary_flow(topic_summaries, coherence_check.suggestions)
        
        # Stage 3: Final Integration
        print("Stage 3: Synthesizing final summary...")
        final_summary = self.synthesize_final_summary(chapter_text, topic_analysis, topic_summaries)
        
        # Create result package
        result = {
            'chapter_id': chapter_id,
            'book_title': book_title,
            'chapter_title': chapter_title,
            'final_summary': final_summary,
            'topic_breakdown': topic_summaries,
            'analysis_metadata': {
                'topics': topic_analysis.topics,
                'themes': topic_analysis.overarching_themes,
                'narrative_flow': topic_analysis.narrative_flow,
                'validation_score': validation.completeness_score,
                'coherence_score': coherence_check.coherence_score
            },
            'timestamp': datetime.now().isoformat()
        }
        
        # Store in FAISS vector store
        print("Storing in vector database...")
        self.vector_store.add_chapter(chapter_id, final_summary, result)
        
        print(f"Chapter {chapter_id} processing complete!")
        return result
    
    def search_similar_chapters(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for similar chapters using semantic similarity
        """
        return self.vector_store.search_similar_chapters(query, k)
    
    def get_chapter_summary(self, chapter_id: str) -> Optional[Dict[str, Any]]:
        """
        Retrieve a specific chapter summary by ID
        """
        return self.vector_store.get_chapter_by_id(chapter_id)
    
    def get_all_chapter_ids(self) -> List[str]:
        """
        Get list of all processed chapter IDs
        """
        return self.vector_store.get_all_chapters()
    
    def process_multiple_chapters(self, chapters: List[Dict[str, str]], 
                                book_title: str = "") -> List[Dict[str, Any]]:
        """
        Batch processing for multiple chapters
        """
        results = []
        total_chapters = len(chapters)
        
        for i, chapter in enumerate(chapters):
            print(f"\n=== Processing Chapter {i+1}/{total_chapters} ===")
            
            try:
                chapter_summary = self.summarize_chapter(
                    chapter_text=chapter['text'],
                    chapter_id=chapter.get('id', f"chapter_{i+1}"),
                    book_title=book_title,
                    chapter_title=chapter.get('title', f"Chapter {i+1}")
                )
                results.append(chapter_summary)
            except Exception as e:
                print(f"Error processing chapter {i+1}: {e}")
                # Continue with next chapter
                continue
        
        return results
    
    def get_database_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the vector database
        """
        return self.vector_store.get_stats()
    
    def _parse_fallback_analysis(self, response: str) -> TopicAnalysis:
        """
        Fallback parser for malformed JSON responses
        """
        return TopicAnalysis(
            topics=[{
                "name": "General Topic", 
                "description": "Extracted from response", 
                "importance_level": "Primary", 
                "relationships": [], 
                "merge_potential": "Standalone"
            }],
            overarching_themes=["General Theme"],
            narrative_flow="Standard narrative progression",
            importance_hierarchy={"Primary": ["General Topic"], "Secondary": [], "Supporting": []}
        )
    
    def _create_default_analysis(self) -> TopicAnalysis:
        """
        Create default analysis when extraction fails
        """
        return TopicAnalysis(
            topics=[{
                "name": "Chapter Content", 
                "description": "Main chapter content", 
                "importance_level": "Primary", 
                "relationships": [], 
                "merge_potential": "Standalone"
            }],
            overarching_themes=["Main Theme"],
            narrative_flow="Chapter presents key information",
            importance_hierarchy={"Primary": ["Chapter Content"], "Secondary": [], "Supporting": []}
        )

# Usage Example
if __name__ == "__main__":
    # Initialize pipeline with Google models
    pipeline = ChapterSummarizationPipeline(
        index_path="./faiss_chapter_index",
        emb_model_name="models/embedding-001",
        llm_model_name="gemini-2.0-flash"
    )
    
    # Example chapter data
    sample_chapters = [
        {
            'id': 'ch1',
            'title': 'Introduction to Machine Learning',
            'text': '''
            Machine learning represents a fundamental shift in how we approach problem-solving 
            in computer science. Rather than explicitly programming solutions, we enable computers 
            to learn patterns from data and make predictions or decisions based on these patterns.
            
            This chapter introduces the core concepts of machine learning, including supervised 
            learning, unsupervised learning, and reinforcement learning. We explore how algorithms 
            can automatically improve their performance through experience, and discuss the 
            mathematical foundations that make this possible.
            
            The applications of machine learning span across industries, from healthcare and 
            finance to entertainment and transportation. Understanding these fundamentals is 
            crucial for anyone looking to work with modern AI systems.
            '''
        },
        {
            'id': 'ch2', 
            'title': 'Data Preprocessing',
            'text': '''
            Before any machine learning algorithm can be applied effectively, the data must be 
            properly prepared and cleaned. This preprocessing stage is often the most time-consuming 
            part of any machine learning project, yet it is crucial for achieving good results.
            
            This chapter covers essential preprocessing techniques including data cleaning, 
            handling missing values, feature scaling, and feature selection. We discuss how 
            to identify and deal with outliers, and explore various methods for transforming 
            categorical variables into numerical representations.
            
            The quality of your data directly impacts the performance of your models. By mastering 
            these preprocessing techniques, you ensure that your algorithms have the best possible 
            foundation for learning meaningful patterns.
            '''
        }
    ]
    
    # Process single chapter
    single_result = pipeline.summarize_chapter(
        chapter_text=sample_chapters[0]['text'],
        chapter_id=sample_chapters[0]['id'],
        book_title="Machine Learning Fundamentals",
        chapter_title=sample_chapters[0]['title']
    )
    
    print("\n=== Single Chapter Result ===")
    print(f"Chapter ID: {single_result['chapter_id']}")
    print(f"Summary: {single_result['final_summary'][:200]}...")
    
    # Process multiple chapters
    batch_results = pipeline.process_multiple_chapters(
        sample_chapters, 
        book_title="Machine Learning Fundamentals"
    )
    
    print(f"\n=== Batch Processing Complete ===")
    print(f"Processed {len(batch_results)} chapters")
    
    # Search for similar chapters
    similar = pipeline.search_similar_chapters("machine learning algorithms and data", k=2)
    print(f"\n=== Similar Chapters Found ===")
    for result in similar:
        print(f"Chapter: {result['chapter_id']}, Score: {result['similarity_score']:.3f}")
    
    # Get database statistics
    stats = pipeline.get_database_stats()
    print(f"\n=== Database Statistics ===")
    print(f"Total chapters: {stats['total_chapters']}")
    print(f"Active chapters: {stats['active_chapters']}")
    
    print("\nPipeline execution complete!")