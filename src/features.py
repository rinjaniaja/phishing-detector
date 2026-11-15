import re
from urllib.parse import urlparse
import tldextract

# pakai daftar bawaan, tidak perlu internet
_ext = tldextract.TLDExtract(suffix_list_urls=())


def extract_features(url):
    url = str(url).strip()
    parsed = urlparse(url if "://" in url else "http://" + url)
    host = parsed.netloc.split(":")[0]
    ext = _ext(url)
    digits = sum(c.isdigit() for c in url)

    return {
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(parsed.path),
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_digits": digits,
        "digit_ratio": digits / max(len(url), 1),
        "num_special": len(re.findall(r"[^\w]", url)),
        "has_at": int("@" in url),
        "has_ip": int(bool(re.fullmatch(r"\d{1,3}(\.\d{1,3}){3}", host))),
        "num_subdomains": len([s for s in ext.subdomain.split(".") if s]),
        "is_https": int(parsed.scheme == "https"),
        "has_query": int(bool(parsed.query)),
        "num_params": len(parsed.query.split("&")) if parsed.query else 0,
    }