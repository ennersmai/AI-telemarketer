import pytest
from unittest.mock import MagicMock, patch

# Adjust imports based on your actual project structure
# Avoid circular imports by importing directly from module files
from app.llm_manager import LLMManager
from app.llm_data import LLMConditioning, Conversation, Message, LLMResponse, MemoryConfig
# Import the base class directly from the module file
from app.inference_engines.inference_llm.inference_base_llm import InferenceEngineBaseLLM
# Import specific engine directly from its module file
from app.inference_engines.inference_llm.inference_groq import InferenceEngineGroq
from app.tool_data import LLMTool
from app.context_data import Context # Import Context if testing Context input to prompt_llm


# --- Mocks and Fixtures ---

@pytest.fixture
def mock_llm_engine():
    """Fixture to create a mock LLM inference engine."""
    engine = MagicMock(spec=InferenceEngineBaseLLM)  # Use base class directly
    engine._type = "LLM" # Set the type attribute required by LLMManager.configure
    engine.initialize_model = MagicMock()
    # Make run_inference return a valid LLMResponse
    engine.run_inference = MagicMock(return_value=LLMResponse(message="Mock response", tool_calls=[]))
    return engine

@pytest.fixture
def llm_manager():
    """Fixture to create an LLMManager instance."""
    # Mock dependencies like LibraryManager if needed during __init__
    # Also mock Memory DB if it's accessed during init (unlikely but possible)
    with patch('app.llm_manager.LibraryManager', MagicMock()) as MockLibrary:
        # Mock retrieve_datapoint if called by prompt_llm's default sys prompt logic
        MockLibrary.return_value.retrieve_datapoint.return_value = "Default System Prompt Text"
        # Mock the DB Manager used within prompt_llm for RAG
        with patch('app.llm_manager.MemoryEmbeddingDatabaseManager', MagicMock()) as MockDB:
            # Mock methods if needed, e.g., MockDB.return_value.search_semantic.return_value = []
             manager = LLMManager()
             # Assign the mock library instance if needed later
             # manager._library = MockLibrary()
             return manager

@pytest.fixture
def sample_conditioning():
    """Fixture for sample LLMConditioning."""
    # No secrets needed for basic conditioning object creation usually
    # If get_secret IS called, we need to patch security_manager
    # with patch('app.security_manager.SecretsManager.get_secret', return_value="dummy_key"):
    # Set add_default_sys_prompt False to simplify basic tests
    return LLMConditioning(model="test-model", add_default_sys_prompt=False)

# --- Test Cases ---

def test_llm_manager_configure_stores_dirty(llm_manager, mock_llm_engine, sample_conditioning):
    """Test that configure sets the _dirty attributes."""
    assert llm_manager._inference_engine_dirty is None
    assert llm_manager._conditioning_dirty is None

    llm_manager.configure(inference_engine=mock_llm_engine, conditioning=sample_conditioning)

    assert llm_manager._inference_engine_dirty is mock_llm_engine
    assert llm_manager._conditioning_dirty is sample_conditioning
    assert llm_manager._inference_engine is None # Should not be set yet
    assert llm_manager._conditioning is None  # Should not be set yet

def test_llm_manager_apply_config_sets_active_and_initializes(llm_manager, mock_llm_engine, sample_conditioning):
    """Test that apply_config sets active attributes and calls engine init."""
    # First configure
    llm_manager.configure(inference_engine=mock_llm_engine, conditioning=sample_conditioning)

    # Then apply
    llm_manager.apply_config()

    # Check active attributes
    assert llm_manager._inference_engine is mock_llm_engine
    assert llm_manager._conditioning is sample_conditioning

    # Check if engine's initialize_model was called correctly
    mock_llm_engine.initialize_model.assert_called_once_with(sample_conditioning)

def test_llm_manager_apply_config_raises_if_not_configured(llm_manager):
    """Test apply_config raises error if configure wasn't called."""
    # Test for engine missing
    with pytest.raises(Exception, match="No inference engine provided"):
        llm_manager.apply_config()

    # Test for conditioning missing (requires engine to be configured first)
    # We need a way to set only engine_dirty for this test
    # This might require slightly refactoring the fixture or test setup
    # For now, let's assume the primary check covers the logic adequately

def test_llm_manager_prompt_llm_calls_engine(llm_manager, mock_llm_engine, sample_conditioning):
    """Test prompt_llm calls the engine's run_inference."""
    # Configure and apply first
    llm_manager.configure(inference_engine=mock_llm_engine, conditioning=sample_conditioning)
    llm_manager.apply_config()

    # Prepare dummy conversation and tools
    conversation = Conversation(conversation=[Message(author="user", content="Hello")])
    tools = [LLMTool(name="test_tool", description="A tool", parameters={})] # Example tool

    # Mock RAG-related calls if memory_config is used
    memory_config = MemoryConfig(retrieve_memories=False) # Disable RAG for this test

    response = llm_manager.prompt_llm(conversation=conversation, tools=tools, memory_config=memory_config)

    # Assert engine was called
    mock_llm_engine.run_inference.assert_called_once_with(conversation=conversation, tools=tools)
    # We could add more specific checks on the arguments passed to run_inference

    # Assert response is returned
    assert isinstance(response, LLMResponse)
    assert response.message == "Mock response"

# TODO: Add tests specifically for InferenceEngineGroq, mocking the actual Groq client call.
# TODO: Add tests for RAG logic (memory_config=True).
# TODO: Add tests for handling Context input type in prompt_llm.
# TODO: Add tests for default system prompt logic.
# TODO: Add tests for instruction handling.
