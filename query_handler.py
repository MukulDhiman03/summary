from langchain.chains import RetrievalQA, LLMChain
from langchain import PromptTemplate
import json

template_dict = {
    "qa": PromptTemplate(template="""
                                        Use the following pieces of information to provide a brief and to-the-point answer to the user's question.
                                        If you don't know the answer, just say that you don't know — don't make one up.

                                        Context: {context}
                                        Question: {question}

                                        In your answer, include a short, simple example that explains the concept in layman's terms.
                                        For example, if the topic is about how a search engine works, you might say:
                                        "It's like looking for a book in a huge library using a card catalog — the engine helps you find the exact shelf and book quickly."

                                        Only return the answer in JSON format defined below and nothing else.

                                        Output Format:
                                        {{
                                        "helpful_answer": "Answer here",
                                        "example": "Layman example here"
                                        }}
                                        """,
                         input_variables=["context", "question"]
                        ),

    "summary": PromptTemplate(template="""
                                    You are given the full content of a document. Your task is to create a concise and clear summary that captures its key points.

                                    Context: {context}

                                    Write the summary in plain language that a non-expert can understand. Focus on the main ideas and important information. Do not include any examples or analogies.

                                    Only return the summary in the JSON format below and nothing else.

                                    Output Format:
                                    {{
                                    "summary": "Short summary here"
                                    }}
                                    """,
                              input_variables=["context"]
                            ),

    "topics": PromptTemplate(template="""
                                        You are given the full content of a chapter. Your task is to extract a list of the key topic names or main headings that summarize what this chapter covers.

                                        Context: {context}

                                        Return only the most relevant and meaningful topic names in the order they appear or make logical sense. Do not include explanations, descriptions, or examples.

                                        Only return the list in the JSON format below and nothing else.

                                        Output Format:
                                        {{
                                        "topics": [
                                        "Topic 1",
                                        "Topic 2",
                                        "Topic 3"
                                        ]
                                        }}
                                        """,
                             input_variables=["context"]
                             )
}


class PromptQueryHandler:
    def __init__(self, llm, retriever, templates):
        self.llm = llm
        self.retriever = retriever
        self.templates = templates

    def _select_prompt(self, mode):
        if mode not in self.templates:
            raise ValueError(f"Unsupported mode: {mode}. Choose from: {list(self.templates.keys())}")
        return self.templates[mode]

    def build_chain(self, mode):
        prompt = self._select_prompt(mode)

        if mode == "qa":
            # For QA, use RetrievalQA which handles retrieval automatically
            return RetrievalQA.from_chain_type(
                llm=self.llm,
                retriever=self.retriever,
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt}
            )
        else:
            # For summary and topics, use LLMChain with full context
            return LLMChain(llm=self.llm, prompt=prompt)

    def get_full_context(self):
        """Get all docs from FAISS docstore."""
        try:
            # Check if retriever has vectorstore and docstore
            if not hasattr(self.retriever, 'vectorstore'):
                raise AttributeError("Retriever does not have vectorstore attribute")

            if not hasattr(self.retriever.vectorstore, 'docstore'):
                raise AttributeError("Vectorstore does not have docstore attribute")

            # Try to get all documents from docstore
            if hasattr(self.retriever.vectorstore.docstore, '_dict'):
                all_docs = list(self.retriever.vectorstore.docstore._dict.values())
            else:
                # Alternative method if _dict is not available
                # Get a large number of documents using similarity search
                all_docs = self.retriever.get_relevant_documents("", k=1000)

            if not all_docs:
                raise ValueError("No documents found in the docstore")

            full_text = "\n\n".join(doc.page_content for doc in all_docs if hasattr(doc, 'page_content'))
            return full_text

        except Exception as e:
            raise RuntimeError(f"Failed to retrieve full context: {e}")

    def get_query(self, question=None, mode="qa"):
        """
        Execute a query based on the specified mode.

        Args:
            question (str): The question to ask (required for 'qa' mode, ignored for others)
            mode (str): The mode of operation ('qa', 'summary', or 'topics')

        Returns:
            dict: Parsed JSON response from the LLM
        """
        if mode not in self.templates:
            raise ValueError(f"Unsupported mode: {mode}. Choose from: {list(self.templates.keys())}")

        chain = self.build_chain(mode)

        if mode == "qa":
            if not question:
                raise ValueError("Question is required for 'qa' mode")

            # For QA mode, RetrievalQA handles retrieval automatically
            result = chain({"query": question})
            output = result.get("result", "")

        else:
            # For summary and topics modes, get full context and use LLMChain
            try:
                context = self.get_full_context()
                result = chain.run(context=context)
                output = result
            except Exception as e:
                return {"error": f"Failed to process {mode} request: {str(e)}"}

        # Safe JSON parsing instead of eval()
        try:
            if isinstance(output, str):
                # Clean the output to extract JSON
                output = output.strip()

                # Try to find JSON content between braces
                start_idx = output.find('{')
                end_idx = output.rfind('}') + 1

                if start_idx != -1 and end_idx > start_idx:
                    json_str = output[start_idx:end_idx]
                    return json.loads(json_str)
                else:
                    # If no JSON found, try parsing the whole string
                    return json.loads(output)
            else:
                return output

        except json.JSONDecodeError as e:
            return {
                "error": f"Failed to parse JSON output: {str(e)}",
                "raw_output": output
            }
        except Exception as e:
            return {
                "error": f"Unexpected error during parsing: {str(e)}",
                "raw_output": output
            }

    def ask_question(self, question):
        """Convenience method for QA mode."""
        return self.get_query(question=question, mode="qa")

    def get_summary(self):
        """Convenience method for summary mode."""
        return self.get_query(mode="summary")

    def get_topics(self):
        """Convenience method for topics mode."""
        return self.get_query(mode="topics")
    
