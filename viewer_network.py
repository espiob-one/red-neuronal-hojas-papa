"""Local-network HTTP access policy, independent from changing DHCP addresses."""
import ipaddress
from urllib.parse import urlsplit


def address(value):
    ip=ipaddress.ip_address(value.split('%',1)[0])
    return getattr(ip,'ipv4_mapped',None) or ip


def client_allowed(peer,allowed_client=None,allowed_subnet=None):
    try:
        ip=address(peer)
        if ip.is_multicast or ip.is_unspecified:
            return False
        if ip.is_loopback:
            return True
        if allowed_client and ip==address(allowed_client):
            return True
        if allowed_subnet=='auto':
            return ip.is_private and not ip.is_reserved
        if allowed_subnet:
            network=ipaddress.ip_network(allowed_subnet,strict=False)
            return network.is_private and ip in network
    except ValueError:
        return False
    return False


def host_allowed(raw,local_ip,port,hostnames=()):
    try:
        parsed=urlsplit('//'+(raw or ''))
        if parsed.username or parsed.password or parsed.path or parsed.query or parsed.fragment:
            return False
        if parsed.port!=port:
            return False
        host=(parsed.hostname or '').lower().rstrip('.')
        allowed={h.lower().rstrip('.') for h in hostnames}|{'localhost','127.0.0.1','::1',local_ip.lower().split('%',1)[0]}
        return host in allowed
    except ValueError:
        return False


def same_origin(origin,host):
    # Browsers send Origin for form POSTs. CLI clients may omit it.
    return origin is None or origin=='http://'+host
