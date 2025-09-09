import requests
import os


def check_and_create_collection(
    collection_name, token_credentials, vector_size=3072, distance="Cosine"
):
    """
    Check if a Qdrant collection exists and create it if it doesn't.

    Args:
        collection_name (str): Name of the collection to check/create
        vector_size (int): Size of the vectors in the collection (default: 3072)
        distance (str): Distance metric for similarity search (default: "Cosine")

    Returns:
        dict: Response from the Qdrant API
    """
    target_audience = os.getenv("TARGET_AUDIENCE")
    print(f"Checking/creating collection: {collection_name}")
    print(f"Target audience: {target_audience}")
    print(f"Vector size: {vector_size}, Distance: {distance}")

    headers = {"Content-Type": "application/json"}

    # Add authorization if token is available
    if token_credentials:
        headers["Authorization"] = f"Bearer {token_credentials.token}"
        print("Authorization header added")
    else:
        print("No token credentials provided")

    # First, check if collection exists
    check_url = f"{target_audience}/collections/{collection_name}"
    print(f"Checking collection at: {check_url}")

    try:
        response = requests.get(check_url, headers=headers)
        print(f"Collection check response status: {response.status_code}")

        if response.status_code == 200:
            print(f"Collection '{collection_name}' already exists")
            collection_info = response.json()
            print(f"Collection info: {collection_info}")
            return collection_info
        else:
            print(f"Collection check failed: {response.text}")
    except requests.exceptions.RequestException as e:
        print(f"Error checking collection: {e}")
        return None

    # If collection doesn't exist, create it
    create_url = f"{target_audience}/collections/{collection_name}"
    create_payload = {"vectors": {"size": vector_size, "distance": distance}}

    print(f"Creating collection at: {create_url}")
    print(f"Create payload: {create_payload}")

    try:
        response = requests.put(create_url, headers=headers, json=create_payload)
        print(f"Collection creation response status: {response.status_code}")
        print(f"Collection creation response: {response.text}")
        response.raise_for_status()
        print(f"Collection '{collection_name}' created successfully")
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error creating collection: {e}")
        return None


def search_qdrant(
    query_vector, token_credentials, collection_name, top=3, qdrant_filter=None
):
    """
    Search Qdrant with optional payload filter.
    """
    target_audience = os.getenv("TARGET_AUDIENCE")
    headers = {
        "Authorization": f"Bearer {token_credentials.token}",
        "Content-Type": "application/json",
    }
    payload = {"vector": query_vector, "top": top, "with_payload": True}
    if qdrant_filter:
        payload["filter"] = qdrant_filter
        print(f"Qdrant filter being sent: {qdrant_filter}")  # Add this debug line
    
    print(f"Full payload being sent: {payload}")  # Add this debug line
    
    response = requests.post(
        f"{target_audience}/collections/{collection_name}/points/search",
        headers=headers,
        json=payload,
    )
    
    # Add better error handling
    if response.status_code != 200:
        print(f"Qdrant search error: {response.status_code}")
        print(f"Error response: {response.text}")
    
    response.raise_for_status()
    return response.json().get("result", [])