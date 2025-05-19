"""API module for EpitopeGraph with versioned endpoints and robust error handling."""

import json
import logging
from functools import wraps
from typing import Any, Callable, Dict, Optional, Union

import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
from requests_cache import CachedSession

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API version
API_VERSION = "v1"
BASE_URL = f"https://api.epitopegraph.org/{API_VERSION}"

# HTTP status codes
HTTP_OK = 200
HTTP_BAD_REQUEST = 400
HTTP_UNAUTHORIZED = 401
HTTP_NOT_FOUND = 404
HTTP_RATE_LIMIT = 429
HTTP_SERVER_ERROR = 500

class APIError(Exception):
    """Base exception for API errors."""
    def __init__(self, message: str, status_code: int, response: Optional[Dict] = None):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)

class BadRequestError(APIError):
    """Raised for 400 errors."""
    pass

class UnauthorizedError(APIError):
    """Raised for 401 errors."""
    pass

class NotFoundError(APIError):
    """Raised for 404 errors."""
    pass

class RateLimitError(APIError):
    """Raised for 429 errors."""
    pass

class ServerError(APIError):
    """Raised for 500 errors."""
    pass

def create_session(cache: bool = True, cache_expiry: int = 3600) -> requests.Session:
    """Create a requests session with retry logic and optional caching.
    
    Args:
        cache: Whether to enable response caching
        cache_expiry: Cache expiry time in seconds
        
    Returns:
        Configured requests session
    """
    if cache:
        session = CachedSession(
            'epitopegraph_cache',
            expire_after=cache_expiry,
            backend='sqlite'
        )
    else:
        session = requests.Session()
    
    # Configure retry strategy
    retry_strategy = Retry(
        total=3,  # number of retries
        backoff_factor=1,  # wait 1, 2, 4 seconds between retries
        status_forcelist=[429, 500, 502, 503, 504],  # HTTP status codes to retry on
    )
    
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    
    return session

def handle_response(response: requests.Response) -> Dict:
    """Handle API response and raise appropriate exceptions.
    
    Args:
        response: Response object from requests
        
    Returns:
        JSON response data
        
    Raises:
        APIError: For various HTTP error codes
    """
    try:
        data = response.json()
    except json.JSONDecodeError:
        data = {"message": response.text}
    
    if response.status_code == HTTP_OK:
        return data
    
    error_map = {
        HTTP_BAD_REQUEST: BadRequestError,
        HTTP_UNAUTHORIZED: UnauthorizedError,
        HTTP_NOT_FOUND: NotFoundError,
        HTTP_RATE_LIMIT: RateLimitError,
        HTTP_SERVER_ERROR: ServerError,
    }
    
    error_class = error_map.get(response.status_code, APIError)
    raise error_class(
        message=data.get("message", "Unknown error"),
        status_code=response.status_code,
        response=data
    )

def api_endpoint(endpoint: str) -> Callable:
    """Decorator for API endpoint functions.
    
    Args:
        endpoint: API endpoint path
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Dict:
            session = create_session()
            url = f"{BASE_URL}/{endpoint}"
            
            try:
                response = session.request(
                    method=func.__name__.upper(),
                    url=url,
                    params=kwargs.get("params"),
                    json=kwargs.get("json"),
                    headers=kwargs.get("headers", {})
                )
                return handle_response(response)
            except requests.RequestException as e:
                logger.error(f"Request failed: {str(e)}")
                raise APIError(
                    message=f"Request failed: {str(e)}",
                    status_code=HTTP_SERVER_ERROR
                )
            finally:
                session.close()
        
        return wrapper
    return decorator

class EpitopeGraphAPI:
    """Client for EpitopeGraph API."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize API client.
        
        Args:
            api_key: Optional API key for authenticated requests
        """
        self.headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
    
    @api_endpoint("epitopes/predict")
    def predict_epitopes(
        self,
        uniprot_id: str,
        distance_cutoff: float = 8.0,
        include_ss: bool = True,
        include_sasa: bool = True
    ) -> Dict:
        """Predict epitopes for a protein.
        
        Args:
            uniprot_id: UniProt accession ID
            distance_cutoff: Distance cutoff for graph edges (Å)
            include_ss: Whether to include secondary structure
            include_sasa: Whether to include SASA
            
        Returns:
            Prediction results
        """
        return {
            "params": {
                "uniprot_id": uniprot_id,
                "distance_cutoff": distance_cutoff,
                "include_ss": include_ss,
                "include_sasa": include_sasa
            }
        }
    
    @api_endpoint("epitopes/batch")
    def predict_batch(
        self,
        uniprot_ids: list[str],
        **kwargs
    ) -> Dict:
        """Batch predict epitopes for multiple proteins.
        
        Args:
            uniprot_ids: List of UniProt accession IDs
            **kwargs: Additional prediction parameters
            
        Returns:
            Batch prediction results
        """
        return {
            "json": {
                "uniprot_ids": uniprot_ids,
                **kwargs
            }
        }
    
    @api_endpoint("structures/fetch")
    def fetch_structure(
        self,
        uniprot_id: str,
        source: str = "alphafold"
    ) -> Dict:
        """Fetch protein structure.
        
        Args:
            uniprot_id: UniProt accession ID
            source: Structure source (alphafold/pdb)
            
        Returns:
            Structure data
        """
        return {
            "params": {
                "uniprot_id": uniprot_id,
                "source": source
            }
        }
    
    @api_endpoint("sequences/fetch")
    def fetch_sequence(
        self,
        uniprot_id: str
    ) -> Dict:
        """Fetch protein sequence.
        
        Args:
            uniprot_id: UniProt accession ID
            
        Returns:
            Sequence data
        """
        return {
            "params": {
                "uniprot_id": uniprot_id
            }
        }
    
    @api_endpoint("benchmarks/run")
    def run_benchmark(
        self,
        dataset: str = "default",
        metrics: list[str] = None
    ) -> Dict:
        """Run benchmark evaluation.
        
        Args:
            dataset: Benchmark dataset name
            metrics: List of metrics to compute
            
        Returns:
            Benchmark results
        """
        return {
            "json": {
                "dataset": dataset,
                "metrics": metrics or ["accuracy", "auc", "f1"]
            }
        } 