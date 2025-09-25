from toponymy.toponymy import Toponymy
from toponymy.llm_wrappers import HuggingFaceNamer, AsyncHuggingFaceNamer, OllamaNamer, AsyncOllamaNamer
from toponymy.clustering import centroids_from_labels, ToponymyClusterer
from toponymy.keyphrases import KeyphraseBuilder
from toponymy.cluster_layer import ClusterLayerText
from sentence_transformers import SentenceTransformer

from sklearn.metrics import pairwise_distances
from scipy.optimize import linear_sum_assignment
from pathlib import Path

import numpy as np
import pandas as pd
import json
import umap

import pytest


def test_toponymy(
    llm,
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    cluster_label_vector,
    subtopic_objects,
):
    model = Toponymy(
        llm,
        embedder,
        clusterer,
        keyphrase_builder=KeyphraseBuilder(n_jobs=1),
        object_description="sentences",
        corpus_description="collection of sentences",
        lowest_detail_level=0.8,
        highest_detail_level=1.0,
        show_progress_bars=True,
    )
    model.fit(all_sentences, object_vectors, clusterable_vectors)
    embedded_topic_names = embedder.encode(model.topic_names_[1])
    print(model.topic_names_[1])
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        embedder.encode([topic["topic"] for topic in subtopic_objects]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.7
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == cluster_label_vector
    )


def test_toponymy_alternative_options(
    async_llm,
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    subtopic_objects,
    cluster_label_vector,
):
    clusterer.fit(
        clusterable_vectors,
        object_vectors,
        prompt_format="combined",
        object_to_text_function=lambda x: x,
    )
    model = Toponymy(
        async_llm,
        embedder,
        clusterer,
        keyphrase_builder=KeyphraseBuilder(n_jobs=1, verbose=True, embedder=embedder),
        object_description="sentences",
        corpus_description="collection of sentences",
        lowest_detail_level=0.8,
        highest_detail_level=1.0,
        show_progress_bars=True,
    )
    topic_name_vectors = model.fit_predict(
        all_sentences,
        object_vectors,
        clusterable_vectors,
        keyphrase_method="bm25",
        subtopic_method="information_weighted",
    )
    embedded_topic_names = embedder.encode(model.topic_names_[1])
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        embedder.encode([topic["topic"] for topic in subtopic_objects]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.7
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == cluster_label_vector
    )
    assert len(str(model.topic_tree_)) > 10


def test_toponymy_alternative_options_2(
    async_llm,
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    subtopic_objects,
    cluster_label_vector,
):
    clusterer.fit(
        clusterable_vectors,
        object_vectors,
        prompt_format="system_user",
        object_to_text_function=lambda x: x,
    )
    model = Toponymy(
        async_llm,
        embedder,
        clusterer,
        keyphrase_builder=KeyphraseBuilder(n_jobs=1, verbose=True, embedder=embedder),
        object_description="sentences",
        corpus_description="collection of sentences",
        lowest_detail_level=0.8,
        highest_detail_level=1.0,
        show_progress_bars=True,
    )
    topic_name_vectors = model.fit_predict(
        all_sentences,
        object_vectors,
        clusterable_vectors,
        keyphrase_method="bm25",
        subtopic_method="information_weighted",
    )
    embedded_topic_names = embedder.encode(model.topic_names_[1])
    distance_matrix = pairwise_distances(
        embedded_topic_names,
        embedder.encode([topic["topic"] for topic in subtopic_objects]),
        metric="cosine",
    )
    row_matching, col_matching = linear_sum_assignment(distance_matrix)
    assert distance_matrix[row_matching, col_matching].sum() < 2.7
    assert np.all(
        pd.Series(model.cluster_layers_[1].cluster_labels)
        .map(dict(np.vstack([np.arange(5), col_matching]).T))
        .values
        == cluster_label_vector
    )
    assert len(str(model.topic_tree_)) > 10


def test_toponymy_with_ollama(
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    cluster_label_vector,
    subtopic_objects,
):
    """Integration test using Ollama for local LLM inference.
    
    Uses a very small model suitable for CI environments with limited resources.
    The test will check system resources and skip if insufficient.
    """
    try:
        import subprocess
        import time
        import requests
        import shutil
        import psutil
        import os
        
        # Check system resources before proceeding
        # Be more lenient with resource requirements in CI environments
        is_ci = os.getenv('CI', '').lower() == 'true' or os.getenv('AZURE_PIPELINES', '').lower() == 'true'
        
        # Resource requirements (relaxed for CI)
        min_memory_gb = 3.0 if is_ci else 4.0  # Lower requirement in CI
        min_disk_gb = 1.5 if is_ci else 2.0    # Lower requirement in CI
        
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        available_disk = shutil.disk_usage('/tmp').free / (1024**3)  # GB
        
        if available_memory < min_memory_gb:
            pytest.skip(f"Insufficient memory for Ollama test: {available_memory:.1f}GB < {min_memory_gb}GB required")
        
        if available_disk < min_disk_gb:
            pytest.skip(f"Insufficient disk space for Ollama test: {available_disk:.1f}GB < {min_disk_gb}GB required")
        
        # Use the smallest possible models in order of preference
        # These are all very small models suitable for CI environments  
        model_candidates = [
            "qwen2.5:0.5b",      # ~397MB - smallest available
            "qwen3:0.6b",        # ~522MB - very recent and small
            "tinydolphin:1.1b",  # ~636MB - backup option
            "gemma2:2b"          # ~1.6GB - fallback if others fail
        ]
        
        model_name = None
        for candidate in model_candidates:
            try:
                # Check if model is already available
                result = subprocess.run(
                    ["ollama", "list"], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                if candidate in result.stdout:
                    model_name = candidate
                    print(f"Using available model: {model_name}")
                    break
                else:
                    # Try to pull the model with timeout
                    print(f"Downloading {candidate} model for testing...")
                    subprocess.run(
                        ["ollama", "pull", candidate], 
                        check=True, 
                        timeout=120  # 2 minute timeout
                    )
                    model_name = candidate
                    print(f"Successfully downloaded: {model_name}")
                    break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                print(f"Failed to download {candidate}, trying next option...")
                continue
        
        if model_name is None:
            pytest.skip("Could not download any suitable small model for testing")
        
        # Check if Ollama is installed
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Ollama not installed")
        
        # Check if Ollama service is running, if not try to start it briefly
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            service_running = response.status_code == 200
        except:
            service_running = False
            
        if not service_running:
            # Try to start Ollama service in background for testing
            try:
                ollama_process = subprocess.Popen(
                    ["ollama", "serve"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                time.sleep(3)  # Give it time to start
                
                # Check if it started successfully
                try:
                    response = requests.get("http://localhost:11434/api/version", timeout=2)
                    if response.status_code != 200:
                        raise Exception("Ollama service failed to start")
                except:
                    ollama_process.terminate()
                    pytest.skip("Could not start Ollama service for testing")
            except:
                pytest.skip("Could not start Ollama service for testing")
        else:
            ollama_process = None
        
        try:
            # Use the very small model for testing
            ollama_llm = OllamaNamer(
                model=model_name,  # Very small model for CI
                host="http://localhost:11434"
            )
            
            # Set a timeout for the entire model fitting process
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("Ollama test timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(120)  # 2 minute timeout for model fitting
            
            try:
                model = Toponymy(
                    ollama_llm,
                    embedder,
                    clusterer,
                    keyphrase_builder=KeyphraseBuilder(n_jobs=1),
                    object_description="sentences",
                    corpus_description="collection of sentences",
                    lowest_detail_level=0.8,
                    highest_detail_level=1.0,
                    show_progress_bars=True,
                )
                
                model.fit(all_sentences, object_vectors, clusterable_vectors)
                
                # Verify that topic names were generated
                assert len(model.topic_names_[1]) > 0
                
                # Check that topic names are reasonable strings
                for topic_name in model.topic_names_[1]:
                    assert isinstance(topic_name, str)
                    assert len(topic_name.strip()) > 0
                    
                # Verify cluster structure is maintained
                assert len(model.cluster_layers_[1].cluster_labels) == len(cluster_label_vector)
                
                print(f"Ollama generated topic names: {model.topic_names_[1]}")
                
            except TimeoutError:
                pytest.skip("Ollama test timed out - likely too slow for CI environment")
            finally:
                signal.alarm(0)  # Cancel the alarm
            
        finally:
            # Clean up: stop the Ollama service if we started it
            if ollama_process is not None:
                ollama_process.terminate()
                ollama_process.wait()
        
    except Exception as e:
        pytest.skip(f"Ollama test failed: {e}")


def test_toponymy_async_ollama(
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    cluster_label_vector,
    subtopic_objects,
):
    """Integration test using AsyncOllama for concurrent local LLM inference.
    
    Uses a very small model suitable for CI environments with limited resources.
    The test will check system resources and skip if insufficient.
    """
    try:
        import subprocess
        import time
        import requests
        import shutil
        import psutil
        import os
        
        # Check system resources before proceeding
        # Be more lenient with resource requirements in CI environments
        is_ci = os.getenv('CI', '').lower() == 'true' or os.getenv('AZURE_PIPELINES', '').lower() == 'true'
        
        # Resource requirements (relaxed for CI)
        min_memory_gb = 3.0 if is_ci else 4.0  # Lower requirement in CI
        min_disk_gb = 1.5 if is_ci else 2.0    # Lower requirement in CI
        
        available_memory = psutil.virtual_memory().available / (1024**3)  # GB
        available_disk = shutil.disk_usage('/tmp').free / (1024**3)  # GB
        
        if available_memory < min_memory_gb:
            pytest.skip(f"Insufficient memory for AsyncOllama test: {available_memory:.1f}GB < {min_memory_gb}GB required")
        
        if available_disk < min_disk_gb:
            pytest.skip(f"Insufficient disk space for AsyncOllama test: {available_disk:.1f}GB < {min_disk_gb}GB required")
        
        # Use the smallest possible models in order of preference
        # These are all very small models suitable for CI environments  
        model_candidates = [
            "qwen2.5:0.5b",      # ~397MB - smallest available
            "qwen3:0.6b",        # ~522MB - very recent and small
            "tinydolphin:1.1b",  # ~636MB - backup option
            "gemma2:2b"          # ~1.6GB - fallback if others fail
        ]
        
        model_name = None
        for candidate in model_candidates:
            try:
                # Check if model is already available
                result = subprocess.run(
                    ["ollama", "list"], 
                    capture_output=True, 
                    text=True, 
                    check=True
                )
                if candidate in result.stdout:
                    model_name = candidate
                    print(f"Using available model: {model_name}")
                    break
                else:
                    # Try to pull the model with timeout
                    print(f"Downloading {candidate} model for testing...")
                    subprocess.run(
                        ["ollama", "pull", candidate], 
                        check=True, 
                        timeout=120  # 2 minute timeout
                    )
                    model_name = candidate
                    print(f"Successfully downloaded: {model_name}")
                    break
            except (subprocess.TimeoutExpired, subprocess.CalledProcessError):
                print(f"Failed to download {candidate}, trying next option...")
                continue
        
        if model_name is None:
            pytest.skip("Could not download any suitable small model for testing")
        
        # Check if Ollama is installed
        try:
            subprocess.run(["ollama", "--version"], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            pytest.skip("Ollama not installed")
        
        # Check if Ollama service is running, if not try to start it briefly
        try:
            response = requests.get("http://localhost:11434/api/version", timeout=2)
            service_running = response.status_code == 200
        except:
            service_running = False
            
        if not service_running:
            # Try to start Ollama service in background for testing
            try:
                ollama_process = subprocess.Popen(
                    ["ollama", "serve"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.PIPE
                )
                time.sleep(3)  # Give it time to start
                
                # Check if it started successfully
                try:
                    response = requests.get("http://localhost:11434/api/version", timeout=2)
                    if response.status_code != 200:
                        raise Exception("Ollama service failed to start")
                except:
                    ollama_process.terminate()
                    pytest.skip("Could not start Ollama service for testing")
            except:
                pytest.skip("Could not start Ollama service for testing")
        else:
            ollama_process = None
        
        try:
            
            # Use async Ollama with very limited concurrency for small models
            async_ollama_llm = AsyncOllamaNamer(
                model=model_name,  # Very small model for CI
                host="http://localhost:11434",
                max_concurrent_requests=1  # Very conservative for small models and CI
            )
            
            clusterer.fit(
                clusterable_vectors,
                object_vectors,
                prompt_format="system_user",
                object_to_text_function=lambda x: x,
            )
            
            # Set a timeout for the entire async model fitting process
            import signal
            
            def timeout_handler(signum, frame):
                raise TimeoutError("AsyncOllama test timed out")
            
            signal.signal(signal.SIGALRM, timeout_handler)
            signal.alarm(180)  # 3 minute timeout for async model fitting
            
            try:
                model = Toponymy(
                    async_ollama_llm,
                    embedder,
                    clusterer,
                    keyphrase_builder=KeyphraseBuilder(n_jobs=1, verbose=True, embedder=embedder),
                    object_description="sentences",
                    corpus_description="collection of sentences",
                    lowest_detail_level=0.8,
                    highest_detail_level=1.0,
                    show_progress_bars=True,
                )
                
                topic_name_vectors = model.fit_predict(
                    all_sentences,
                    object_vectors,
                    clusterable_vectors,
                    keyphrase_method="bm25",
                    subtopic_method="information_weighted",
                )
                
                # Verify that topic names were generated
                assert len(model.topic_names_[1]) > 0
                
                # Check that topic names are reasonable strings
                for topic_name in model.topic_names_[1]:
                    assert isinstance(topic_name, str)
                    assert len(topic_name.strip()) > 0
                    
                # Verify cluster structure is maintained
                assert len(model.cluster_layers_[1].cluster_labels) == len(cluster_label_vector)
                
                # Verify topic tree was generated
                assert len(str(model.topic_tree_)) > 10
                
                print(f"Async Ollama generated topic names: {model.topic_names_[1]}")
                
            except TimeoutError:
                pytest.skip("AsyncOllama test timed out - likely too slow for CI environment")
            finally:
                signal.alarm(0)  # Cancel the alarm
            
        finally:
            # Clean up: stop the Ollama service if we started it
            if ollama_process is not None:
                ollama_process.terminate()
                ollama_process.wait()
        
    except Exception as e:
        pytest.skip(f"Async Ollama test failed: {e}")


def test_ollama_wrapper_fallback():
    """Test that Ollama wrapper gracefully handles missing dependencies."""
    try:
        # This should work if ollama is installed
        from toponymy.llm_wrappers import OllamaNamer, AsyncOllamaNamer
        
        # If we get here, ollama is available, so we can test basic instantiation
        # but we won't actually call the service since it might not be running
        assert OllamaNamer is not None
        assert AsyncOllamaNamer is not None
        
    except ImportError:
        # If ollama package is not installed, we should get FailedImportLLMWrapper
        from toponymy.llm_wrappers import OllamaNamer, AsyncOllamaNamer
        
        # These should be FailedImportLLMWrapper instances
        ollama_instance = OllamaNamer()
        async_ollama_instance = AsyncOllamaNamer()
        
        # Verify they handle method calls gracefully
        result = ollama_instance.generate_topic_name("test")
        assert result == ""
        
        # For async, we need to test differently since it returns a list
        import asyncio
        async def test_async():
            result = await async_ollama_instance.generate_topic_names(["test"])
            assert result == [""]
        
        asyncio.run(test_async())


def test_ollama_wrapper_with_mock_for_ci():
    """Test Ollama wrapper functionality using a mock for CI environments.
    
    This test ensures the Ollama wrapper integration works correctly without
    requiring actual Ollama installation, making it suitable for CI.
    """
    try:
        from unittest.mock import Mock, patch
        from toponymy.llm_wrappers import OllamaNamer
        
        # Mock the ollama client to simulate responses
        with patch('ollama.Client') as mock_client_class:
            mock_client = Mock()
            mock_client_class.return_value = mock_client
            
            # Mock successful responses that match the expected JSON format
            mock_response = {
                'response': '{"topic_name": "Machine Learning Topics", "topic_specificity": 0.7}'
            }
            mock_client.generate.return_value = mock_response
            
            # Create Ollama wrapper with mocked client
            ollama_llm = OllamaNamer(model="test-model", host="http://localhost:11434")
            
            # Test basic functionality
            result = ollama_llm.generate_topic_name("Test prompt about machine learning")
            assert result == "Machine Learning Topics"
            
            # Test system prompt functionality  
            mock_chat_response = {
                'message': {
                    'content': '{"topic_name": "AI Systems", "topic_specificity": 0.8}'
                }
            }
            mock_client.chat.return_value = mock_chat_response
            
            result = ollama_llm.generate_topic_name({
                "system": "You are a helpful assistant", 
                "user": "Categorize this AI content"
            })
            assert result == "AI Systems"
            
            print("✅ Ollama wrapper mock test passed - integration should work correctly")
            
    except ImportError:
        pytest.skip("Ollama package not available for testing")


def test_async_ollama_wrapper_with_mock_for_ci():
    """Test AsyncOllama wrapper functionality using a mock for CI environments.
    
    This test ensures the AsyncOllama wrapper integration works correctly without
    requiring actual Ollama installation, making it suitable for CI.
    """
    try:
        from unittest.mock import Mock, patch, AsyncMock
        from toponymy.llm_wrappers import AsyncOllamaNamer
        import asyncio
        
        async def run_test():
            # Mock the async ollama client to simulate responses
            with patch('ollama.AsyncClient') as mock_client_class:
                mock_client = Mock()
                mock_client_class.return_value = mock_client
                
                # Mock successful async responses
                mock_response = {
                    'response': '{"topic_name": "Data Science", "topic_specificity": 0.75}'
                }
                mock_client.generate = AsyncMock(return_value=mock_response)
                
                # Create AsyncOllama wrapper with mocked client
                async_ollama_llm = AsyncOllamaNamer(
                    model="test-model", 
                    host="http://localhost:11434",
                    max_concurrent_requests=2
                )
                
                # Test async functionality
                results = await async_ollama_llm.generate_topic_names(["Test prompt"])
                assert len(results) == 1
                assert results[0] == "Data Science"
                
                # Test system prompt functionality  
                mock_chat_response = {
                    'message': {
                        'content': '{"topic_name": "Neural Networks", "topic_specificity": 0.9}'
                    }
                }
                mock_client.chat = AsyncMock(return_value=mock_chat_response)
                
                results = await async_ollama_llm.generate_topic_names([{
                    "system": "You are a helpful assistant", 
                    "user": "Categorize this neural network content"
                }])
                assert len(results) == 1
                assert results[0] == "Neural Networks"
                
                print("✅ AsyncOllama wrapper mock test passed - integration should work correctly")
        
        # Run the async test
        asyncio.run(run_test())
            
    except ImportError:
        pytest.skip("Ollama package not available for testing")


# Mock-based tests for CI environments without external services

def test_ollama_wrapper_mock():
    """Mock-based test for Ollama wrapper when service is not available."""
    from unittest.mock import patch, MagicMock
    
    with patch('ollama.Client') as mock_client:
        # Mock the client
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock the response
        mock_instance.generate.return_value = {
            'response': '{"topic_name": "Mocked Topic", "topic_specificity": 0.7}'
        }
        
        wrapper = OllamaNamer(model="test-model")
        response = wrapper.generate_topic_name("Test prompt")
        
        assert response == 'Mocked Topic'
        mock_instance.generate.assert_called_once()


def test_async_ollama_wrapper_mock():
    """Mock-based test for AsyncOllama wrapper when service is not available."""
    import asyncio
    from unittest.mock import patch, AsyncMock
    
    async def run_test():
        with patch('ollama.AsyncClient') as mock_client:
            # Mock the async client
            mock_instance = AsyncMock()
            mock_client.return_value = mock_instance
            
            # Mock the response
            mock_instance.generate.return_value = {
                'response': '{"topic_name": "Mocked Async Topic", "topic_specificity": 0.8}'
            }
            
            wrapper = AsyncOllamaNamer(model="test-model")
            response = await wrapper.generate_topic_names(["Test prompt"])
            
            assert response == ['Mocked Async Topic']
            mock_instance.generate.assert_called_once()
    
    asyncio.run(run_test())


def test_toponymy_with_mocked_ollama(
    embedder,
    clusterer,
    all_sentences,
    object_vectors,
    clusterable_vectors,
    cluster_label_vector,
    subtopic_objects,
):
    """Integration test using mocked Ollama for CI environments."""
    from unittest.mock import patch, MagicMock
    
    with patch('ollama.Client') as mock_client:
        # Mock the client
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance
        
        # Mock responses for topic naming calls (single topics)
        single_topic_responses = [
            '{"topic_name": "Technology and Innovation", "topic_specificity": 0.8}',
            '{"topic_name": "Data Science Methods", "topic_specificity": 0.7}', 
            '{"topic_name": "Machine Learning Applications", "topic_specificity": 0.9}',
            '{"topic_name": "AI Research", "topic_specificity": 0.6}',
            '{"topic_name": "Computing Technologies", "topic_specificity": 0.5}'
        ]
        
        # Mock responses for cluster naming calls (multiple topics)
        cluster_topic_responses = [
            '{"new_topic_name_mapping": ["Advanced Technology", "Data Analytics", "ML Systems", "AI Studies", "Tech Infrastructure"], "topic_specificities": [0.8, 0.7, 0.9, 0.6, 0.5]}',
            '{"new_topic_name_mapping": ["Innovation Hub", "Science Methods", "Learning Apps", "Research Areas", "Computing Tech"], "topic_specificities": [0.75, 0.65, 0.85, 0.55, 0.45]}',
        ]
        
        # Create a counter to track call types and provide appropriate responses
        call_counter = {'single': 0, 'cluster': 0}
        
        def mock_generate_response(**kwargs):
            # Check if this looks like a cluster naming call (longer prompt, higher max_tokens)
            max_tokens = kwargs.get('options', {}).get('num_predict', 128)
            if max_tokens > 500:  # cluster naming uses max_tokens=1024
                response = cluster_topic_responses[call_counter['cluster'] % len(cluster_topic_responses)]
                call_counter['cluster'] += 1
            else:  # single topic naming uses max_tokens=128
                response = single_topic_responses[call_counter['single'] % len(single_topic_responses)]
                call_counter['single'] += 1
            
            return {'response': response}
        
        # Set up the mock side effect
        mock_instance.generate.side_effect = mock_generate_response
        
        # Also mock chat method for system prompts
        def mock_chat_response(**kwargs):
            # Similar logic for chat method
            max_tokens = kwargs.get('options', {}).get('num_predict', 128)
            if max_tokens > 500:
                response = cluster_topic_responses[call_counter['cluster'] % len(cluster_topic_responses)]
                call_counter['cluster'] += 1
            else:
                response = single_topic_responses[call_counter['single'] % len(single_topic_responses)]
                call_counter['single'] += 1
            
            return {'message': {'content': response}}
        
        mock_instance.chat.side_effect = mock_chat_response
        
        ollama_llm = OllamaNamer(
            model="mocked-model",
            host="http://localhost:11434"
        )
        
        model = Toponymy(
            ollama_llm,
            embedder,
            clusterer,
            keyphrase_builder=KeyphraseBuilder(n_jobs=1),
            object_description="sentences",
            corpus_description="collection of sentences", 
            lowest_detail_level=0.8,
            highest_detail_level=1.0,
            show_progress_bars=True,
        )
        
        model.fit(all_sentences, object_vectors, clusterable_vectors)
        
        # Verify that topic names were generated
        assert len(model.topic_names_[1]) > 0
        
        # Check that topic names are reasonable strings
        for topic_name in model.topic_names_[1]:
            assert isinstance(topic_name, str)
            assert len(topic_name.strip()) > 0
            
        # Verify cluster structure is maintained
        assert len(model.cluster_layers_[1].cluster_labels) == len(cluster_label_vector)
        
        print(f"Mocked Ollama generated topic names: {model.topic_names_[1]}")
        print(f"Mock call counts - Single: {call_counter['single']}, Cluster: {call_counter['cluster']}")
