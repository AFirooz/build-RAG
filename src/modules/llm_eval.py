from sklearn.metrics.pairwise import cosine_similarity
from langchain.embeddings import HuggingFaceEmbeddings
import numpy as np



class LLM_Evaluator:
    def __init__(self, model_name: str="sentence-transformers/paraphrase-albert-small-v2"):
        self.embedding_model = HuggingFaceEmbeddings(model_name=model_name)

        self._run_get_docs = False
        self._run_get_answers = False
        
        # Detail-Oriented Questions
        self.detail_query = [
            "What is the purpose of the 'Tachyon' ray tracing feature in VMD?",
            # "What file formats does VMD support for volumetric maps?",
            # "How does VMD enable interactive molecular dynamics simulations?",
            # "What is the default coloring method for molecular structures in VMD?",
            # "Describe the function of the 'Representations' item in the Graphics menu.",
            # "What are the hardware requirements for running VMD in graphics-enabled mode?",
            # "How does VMD utilize GPU acceleration for molecular rendering?",
            # "What scripting languages does VMD support for text command processing?",
            # "What is the role of the 'RMS Fit and Alignment' extension in molecular analysis?",
            # "Explain the functionality of the 'volmap coulomb' feature in VMD."
        ]

        # Abstract Questions
        self.abstract_query = [
            "How does VMD enhance the visualization of dynamic molecular data?",
            # "Discuss the advantages of using GPU acceleration in VMD.",
            # "What are the implications of VMD being written in C and C++ for its extensibility?",
            # "In what ways can VMD contribute to collaborative research in structural biology?",
            # "How does VMD handle stereoscopic visualization for immersive experiences?",
            # "Why is system memory a critical resource for VMD's batch-mode analysis?",
            # "How does VMD's plugin system benefit users working with non-standard file formats?",
            # "What are the potential applications of VMD's volumetric data visualization capabilities?",
            # "Why is the 'Render Window' essential for generating high-quality molecular images?",
            # "How does VMD support the customization of user sessions?"
        ]

        # Coding Questions
        self.coding_query = [
            "How can one combine multiple atom selections in Python within VMD?",
            # "Write a Python script to change the coloring method of a molecule in VMD.",
            # "What Python modules are available within VMD for manipulating atom selections?",
            # "Describe how VMD's vmdnumpy module can be utilized for matrix operations.",
            # "How can Python callbacks be used to automate animations in VMD?"
        ]

        # GUI-Related Questions
        self.gui_query = [
            "What are the primary functions of the Main Window in VMD's GUI?",
            # "How do you use the Graphics Window to manipulate molecular structures in VMD?",
            # "What is the procedure for rendering images via the GUI in VMD?",
            # "How does the Labels Window assist in annotating molecular structures?",
            # "Describe the use of the Mouse Menu in controlling molecule visualization."
        ]
    
    def _format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    def get_docs(self, retriever):
        self.detail_docs = [retriever.get_relevant_documents(q) for q in self.detail_query]
        self.abstract_docs = [retriever.get_relevant_documents(q) for q in self.abstract_query]
        self.coding_docs = [retriever.get_relevant_documents(q) for q in self.coding_query]
        self.gui_docs = [retriever.get_relevant_documents(q) for q in self.gui_query]

        # Format documents for evaluation using mappings
        self.detail_docs = list(map(self._format_docs, self.detail_docs))
        self.abstract_docs = list(map(self._format_docs, self.abstract_docs))
        self.coding_docs = list(map(self._format_docs, self.coding_docs))
        self.gui_docs = list(map(self._format_docs, self.gui_docs))

        self._run_get_docs = True

    def get_answers(self, chain):
        self.detail_answers = [chain.invoke(q) for q in self.detail_query]
        self.abstract_answers = [chain.invoke(q) for q in self.abstract_query]
        self.coding_answers = [chain.invoke(q) for q in self.coding_query]
        self.gui_answers = [chain.invoke(q) for q in self.gui_query]

        # self.detail_answers = list(map(chain.invoke, self.detail_query))
        # self.abstract_answers = list(map(chain.invoke, self.abstract_query))
        # self.coding_answers = list(map(chain.invoke, self.coding_query))
        # self.gui_answers = list(map(chain.invoke, self.gui_query))
        self._run_get_answers = True

    def evaluate_rag_answer(self, question: str = None, answer: str = None, document: str = None) -> float:
        """
        Evaluate a RAG system's answer using LangChain embeddings and FAISS similarity search.

        Args:
            question (str): The input question.
            answer (str): The RAG system's generated answer.
            document (str): The document from which the answer is derived.

        Returns:
            float: A relevance score between 0 and 1.
        """
        # Check if get_docs and get_answers have been run
        if (not self._run_get_docs or not self._run_get_answers) and (question is None or answer is None or document is None):
            raise ValueError("Please run get_docs and get_answers before evaluating.")

        # Generate embeddings for question, answer, and document
        question_embedding = np.array(self.embedding_model.embed_query(question)).reshape(1, -1)
        answer_embedding = np.array(self.embedding_model.embed_query(answer)).reshape(1, -1)
        document_embedding = np.array(self.embedding_model.embed_query(document)).reshape(1, -1)

        # Compute cosine similarities using LangChain's utility
        doc_similarity = cosine_similarity(answer_embedding, document_embedding)[0][0]
        question_similarity = cosine_similarity(answer_embedding, question_embedding)[0][0]

        # Weighted score: 70% document relevance, 30% question relevance
        score = 0.7 * doc_similarity + 0.3 * question_similarity
        return round(score, 3)

    def evaluate_answers_list(self):
        """Evaluate all answers in the list of questions and answers.
        """
        # Evaluate relevance scores for each question-answer pair
        self.llm_score = {"detail": [], "abstract": [], "coding": [], "gui": []}

        for q, a, d in zip(self.detail_query, self.detail_answers, self.detail_docs):
            self.llm_score["detail"].append(self.evaluate_rag_answer(q, a, d))

        for q, a, d in zip(self.abstract_query, self.abstract_answers, self.abstract_docs):
            self.llm_score["abstract"].append(self.evaluate_rag_answer(q, a, d))

        for q, a, d in zip(self.coding_query, self.coding_answers, self.coding_docs):
            self.llm_score["coding"].append(self.evaluate_rag_answer(q, a, d))
        
        for q, a, d in zip(self.gui_query, self.gui_answers, self.gui_docs):
            self.llm_score["gui"].append(self.evaluate_rag_answer(q, a, d))
