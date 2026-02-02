"""Streamlit web application for RAG-based document Q&A."""

import streamlit as st
from pathlib import Path

from config import MISTRAL_API_KEY, VECTORSTORE_DIR, TOP_K_RESULTS
from rag import query_rag, get_collection


# Page configuration
st.set_page_config(
    page_title="浙江省市政工程预算定额辅助查询平台",
    page_icon="🚇",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .source-box {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        border-left: 4px solid #1f77b4;
    }
    .page-badge {
        background-color: #1f77b4;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
    }
    .section-badge {
        background-color: #2ecc71;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.8em;
        margin-left: 5px;
    }
</style>
""", unsafe_allow_html=True)


def check_setup() -> tuple[bool, str]:
    """Check if the system is properly set up."""
    if not MISTRAL_API_KEY:
        return False, "MISTRAL_API_KEY not found. Please set it in your .env file."
    
    vectorstore_path = VECTORSTORE_DIR / "chroma.sqlite3"
    if not vectorstore_path.exists():
        return False, "Vector store not found. Please run `python ingest.py` first to process the document."
    
    return True, "System ready!"


def display_sources(sources: list[dict]):
    """Display retrieved source chunks."""
    st.subheader("📚 检索资源")
    
    for i, source in enumerate(sources, 1):
        meta = source["metadata"]
        start_page = meta.get("start_page", "?")
        end_page = meta.get("end_page", start_page)
        section = meta.get("section", "")
        distance = source.get("distance", 0)
        
        # Page info
        if start_page == end_page:
            page_str = f"Page {start_page}"
        else:
            page_str = f"Pages {start_page}-{end_page}"
        
        # Create expander for each source
        with st.expander(f"Source {i}: {page_str} (relevance: {1-distance:.1%})", expanded=(i == 1)):
            # Badges
            col1, col2, col3 = st.columns([1, 1, 2])
            with col1:
                st.markdown(f"**📄 {page_str}**")
            with col2:
                if section:
                    st.markdown(f"**📑 {section}**")
            with col3:
                st.markdown(f"**Similarity: {1-distance:.1%}**")
            
            # Table header if present
            if meta.get("table_header"):
                st.info(f"📊 Table context: `{meta['table_header'][:100]}...`" if len(meta.get('table_header', '')) > 100 else f"📊 Table context: `{meta.get('table_header')}`")
            
            # Document content
            st.markdown("---")
            st.markdown(source["document"])


def main():
    """Main application."""
    st.title("🚇 浙江省市政工程预算定额辅助查询平台")
    st.markdown("人工智能辅助(AI-assist)检索相关市政工程预算")
    
    # Check setup
    is_ready, status_msg = check_setup()
    
    if not is_ready:
        st.error(f"⚠️ Setup Required: {status_msg}")
        st.markdown("""
        ### Setup Instructions
        1. Create a `.env` file with your Mistral API key:
           ```
           MISTRAL_API_KEY=your_key_here
           ```
        2. Copy `Tunnel budget.pdf` to the `data/` folder
        3. Run the ingestion script:
           ```
           python ingest.py
           ```
        """)
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ 设置")
        
        n_results = st.slider(
            "设置检索资源数量",
            min_value=1,
            max_value=10,
            value=TOP_K_RESULTS,
            help="检索数量越多，答案越丰富，但响应速度也越慢"
        )
        
        st.markdown("---")
        st.header("💡 问题示例")
        
        example_questions = [
            "在路面标线中，纵向标线的工程预算定额是多少?",
            "在现浇混凝土工程中，承台的工程预算定额是多少?",
            "在隧道爆破开挖中，平洞钻爆开挖工程预算定额是多少？当断面面积在100平方米以内?",
            # "What are the safety-related expenditures?",
            # "What is the timeline for the project?",
        ]
        
        for q in example_questions:
            if st.button(q, key=f"example_{q[:20]}"):
                st.session_state.query = q
        
        st.markdown("---")
        st.markdown("### 📊 系统信息")
        try:
            collection = get_collection()
            st.metric("索引块", collection.count())
        except Exception as e:
            st.warning("Could not load collection stats")
    
    # Main content area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Query input
        query = st.text_input(
            "🔍 进行工程预算定额查询:",
            value=st.session_state.get("query", ""),
            # placeholder="e.g., What is the total budget for tunnel ventilation?"
        )
        
        search_button = st.button("🚀 检索", type="primary", use_container_width=True)
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        clear_button = st.button("🗑️ 清除", use_container_width=True)
        if clear_button:
            st.session_state.query = ""
            st.session_state.pop("result", None)
            st.rerun()
    
    # Process query
    if search_button and query:
        with st.spinner("🔍 答案生成中..."):
            try:
                result = query_rag(query, n_results=n_results)
                st.session_state.result = result
            except Exception as e:
                st.error(f"Error processing query: {str(e)}")
                return
    
    # Display results
    if "result" in st.session_state:
        result = st.session_state.result
        
        st.markdown("---")
        
        # Answer section
        st.subheader("💬 搜索结果")
        st.markdown(result["answer"])
        
        st.markdown("---")
        
        # Sources section
        display_sources(result["sources"])


if __name__ == "__main__":
    main()
