from enum import Enum


class ForwardingAlgorithm(Enum):
    NO_FORWARDING = 1  # Baseline ETSI CPS
    GBC = 2  # Geographically-Scoped Broadcast
    MULTI_HOP = 3  # Proposed Application Layer Multi-Hop
