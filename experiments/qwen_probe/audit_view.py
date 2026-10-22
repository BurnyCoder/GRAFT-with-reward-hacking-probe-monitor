"""Global: readable blinded evidence without truncation. Local: share exact repeated messages.

Sources: https://docs.python.org/3.12/library/json.html and /copy.html.
This display never reads treatment mappings, changes packets, or assigns judgments.
"""
from copy import deepcopy  # Local: protect the frozen original from replacement with display references.
import json  # Local: exact canonical message content is the deduplication key, not a similarity score.


def compact_packet(packet):
    """Global: preserve all evidence. Local: replace repeated message objects with ordered block indices."""
    displayed = deepcopy(packet)  # Local: retain every field, including raw outputs and unknown values.
    blocks, indices = [], {}  # Local: one full copy of each exact message, with a stable first-seen index.

    def reference(message):
        """Local: exact object equality permits sharing; different role/tool fields remain distinct."""
        key = json.dumps(message, sort_keys=True, ensure_ascii=False)  # Local: key ordering does not alter JSON meaning.
        if key not in indices:  # Local: never replace near-duplicates or omit text within a message.
            indices[key] = len(blocks)  # Local: indices are zero-based positions in the visible block list.
            blocks.append(message)  # Local: keep the complete copied message, including tools and observations.
        return indices[key]  # Local: repeated occurrences retain their position in each conversation.

    containers = [displayed, *displayed.get("raw_model_calls", [])]  # Local: compact both ATIF and replayed model inputs.
    for container in containers:  # Local: raw responses and all non-message fields remain unchanged.
        if "messages" in container:  # Local: absent evidence stays absent rather than becoming an empty trajectory.
            container["messages"] = [reference(message) for message in container["messages"]]  # Local: retain order and multiplicity.
    return {"message_blocks": blocks, "packet": displayed}  # Local: expanding each message index reconstructs the original packet.
