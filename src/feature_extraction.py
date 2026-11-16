import re
import math
import ipaddress
import tldextract

from urllib.parse import urlparse, parse_qs

def prepare_url(url):
    """
    Prepare a URL for consistent parsing.
    """

    url = str(url).strip()

    if not url:
        return ""

    if "://" not in url and not url.startswith("//"):
        url = "http://" + url

    return url

def get_url_components(url):
    """
    Extract hostname, path, query, and fragment from a URL.
    """

    prepared_url = prepare_url(url)

    parsed_url = urlparse(prepared_url)

    hostname = parsed_url.hostname or ""
    path = parsed_url.path
    query = parsed_url.query
    fragment = parsed_url.fragment

    return hostname, path, query, fragment

def get_domain_info(hostname):
    """
    Extract subdomain, registered domain and suffix.
    """

    import tldextract

    result = tldextract.extract(hostname)

    subdomain_count = (
        len(result.subdomain.split("."))
        if result.subdomain else 0
    )

    domain_length = len(hostname)

    return subdomain_count, domain_length

def get_character_features(url):
    """
    Extract basic character-based features from a URL.
    """

    url = str(url).strip()
    url_lower = url.lower()

    url_length = len(url)
    digit_count = sum(char.isdigit() for char in url)
    special_char_count = len(re.findall(r"[^a-zA-Z0-9\s]", url))

    features = {
        "url_length": url_length,
        "dot_count": url.count("."),
        "hyphen_count": url.count("-"),
        "digit_count": digit_count,
        "special_char_count": special_char_count,
        "digit_ratio": digit_count / url_length if url_length else 0,
        "percent_encoding_count": len(
            re.findall(r"%[0-9a-fA-F]{2}", url)
        ),
        "has_at_symbol": int("@" in url),
        "has_https": int(url_lower.startswith("https://")),
        "has_http": int(url_lower.startswith("http://")),
        "has_fragment": int("#" in url)
    }

    return features

def get_path_query_features(path, query):
    """
    Extract features related to URL path and query.
    """

    path_length = len(path)

    path_segment_count = sum(
        1 for segment in path.split("/") if segment
    )

    query_length = len(query)

    query_param_count = (
        len(parse_qs(query, keep_blank_values=True))
        if query else 0
    )

    features = {
        "path_length": path_length,
        "path_segment_count": path_segment_count,
        "query_length": query_length,
        "query_param_count": query_param_count
    }

    return features

def get_hostname_features(hostname, parsed_url):
    """
    Extract hostname-related security features.
    """

    # Check whether the hostname is an IP address
    try:
        ipaddress.ip_address(hostname)
        has_ip_address = 1
    except ValueError:
        has_ip_address = 0

    # Check for an explicitly specified non-standard port
    try:
        port = parsed_url.port

        has_nonstandard_port = int(
            port is not None and port not in [80, 443]
        )

    except ValueError:
        has_nonstandard_port = 0

    features = {
        "has_ip_address": has_ip_address,
        "has_nonstandard_port": has_nonstandard_port
    }

    return features

def get_suspicious_features(hostname, url):
    """
    Extract suspicious keyword and URL shortening features.
    """

    suspicious_keywords = [
        "login",
        "verify",
        "account",
        "update",
        "secure",
        "banking",
        "signin",
        "password",
        "confirm",
        "wallet",
        "support",
        "authentication"
    ]

    shortening_services = [
        "bit.ly",
        "tinyurl.com",
        "t.co",
        "goo.gl",
        "ow.ly",
        "is.gd",
        "buff.ly",
        "adf.ly",
        "cutt.ly",
        "shorturl.at"
    ]

    url_lower = url.lower()
    hostname_lower = hostname.lower()

    suspicious_keyword_count = sum(
        url_lower.count(keyword)
        for keyword in suspicious_keywords
    )

    has_shortening_service = int(
        any(
            hostname_lower == service
            or hostname_lower.endswith("." + service)
            for service in shortening_services
        )
    )

    return {
        "suspicious_keyword_count": suspicious_keyword_count,
        "has_shortening_service": has_shortening_service
    }

def get_domain_info(hostname):
    """
    Extract domain length and subdomain count.
    """

    result = tldextract.extract(hostname)

    subdomain_count = (
        len(result.subdomain.split("."))
        if result.subdomain else 0
    )

    domain_length = len(hostname)

    return {
        "domain_length": domain_length,
        "subdomain_count": subdomain_count
    }

def extract_features(url):
    """
    Extract all selected features from a URL.
    """

    original_url = str(url).strip()

    if not original_url:
        raise ValueError("URL cannot be empty.")

    prepared_url = prepare_url(original_url)

    try:
        parsed_url = urlparse(prepared_url)
        hostname = parsed_url.hostname or ""
        parsed_url.port

    except ValueError:
        raise ValueError("Invalid URL or port.")

    if not hostname or any(char.isspace() for char in hostname):
        raise ValueError("Invalid hostname.")

    path = parsed_url.path
    query = parsed_url.query

    features = {}

    features.update(get_character_features(original_url))
    features.update(get_path_query_features(path, query))
    features.update(get_domain_info(hostname))
    features.update(get_hostname_features(hostname, parsed_url))
    features.update(get_suspicious_features(hostname, original_url))

    return features