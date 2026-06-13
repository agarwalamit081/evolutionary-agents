"""Shared network-safety helpers for URL-fetching builtin tools.

Guards against Server-Side Request Forgery (SSRF): blocks requests to private,
loopback, and link-local hosts so a model-supplied URL cannot be used to reach
internal services or cloud metadata endpoints.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Optional
from urllib.parse import urlparse


# IPv4 networks an agent must never be pointed at.
_BLOCKED_V4_NETWORKS = (
    ipaddress.ip_network("0.0.0.0/8"),  # "this host"
    ipaddress.ip_network("10.0.0.0/8"),  # private (RFC1918)
    ipaddress.ip_network("100.64.0.0/10"),  # CGNAT
    ipaddress.ip_network("127.0.0.0/8"),  # loopback
    ipaddress.ip_network("169.254.0.0/16"),  # link-local (incl. cloud metadata)
    ipaddress.ip_network("172.16.0.0/12"),  # private (RFC1918)
    ipaddress.ip_network("192.168.0.0/16"),  # private (RFC1918)
    ipaddress.ip_network("224.0.0.0/4"),  # multicast
)


def assert_public_host(url: str) -> Optional[str]:
    """Return an error string if ``url`` points at a non-public host.

    Resolves the hostname and rejects loopback, private, link-local,
    multicast, and unspecified addresses (both IPv4 and IPv6). Returns
    ``None`` when the host is safe to fetch.

    Note: this is a *blocking* DNS resolution — callers running on the event
    loop should invoke it via ``asyncio.to_thread``.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return f"ERROR: Only http/https URLs are allowed, got scheme '{parsed.scheme}'"
    host = parsed.hostname
    if not host:
        return f"ERROR: URL has no hostname: {url}"

    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return f"ERROR: Cannot resolve host: {host}"

    for info in infos:
        addr = info[4][0]
        try:
            ip = ipaddress.ip_address(addr)
        except ValueError:
            continue
        if ip.is_loopback or ip.is_link_local or ip.is_multicast or ip.is_unspecified:
            return f"ERROR: Blocked non-public host: {host} ({ip})"
        if ip.version == 4 and any(ip in net for net in _BLOCKED_V4_NETWORKS):
            return f"ERROR: Blocked private network: {host} ({ip})"
        if ip.version == 6 and (ip.is_private or ip.is_site_local):
            return f"ERROR: Blocked private network: {host} ({ip})"
    return None
