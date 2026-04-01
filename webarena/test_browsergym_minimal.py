#!/usr/bin/env python
"""
Test script for action normalization functionality.
"""

from memory_agents.utils.utils import parse_action, extract_element_description, normalize_action, calculate_action_similarity

def test_parse_action():
    """Test action parsing."""
    print("\n=== Testing parse_action ===")

    test_cases = [
        "select_option('1240', 'Price')",
        "fill('260', 'switch card holder')",
        "click('223')",
        "scroll(0, 500)",
        "send_msg_to_user('Task completed')"
    ]

    for action in test_cases:
        result = parse_action(action)
        print(f"\nAction: {action}")
        print(f"  Type: {result['type']}")
        print(f"  BID: {result['bid']}")
        print(f"  Params: {result['params']}")


def test_extract_element_description():
    """Test element description extraction."""
    print("\n\n=== Testing extract_element_description ===")

    # Example state from accessibility tree
    state_text = """
    [1240] combobox 'Sort by', expanded=False
    [1241] button 'Search'
    [260] textbox 'Card Holder Name', required=True
    [223] link 'My Account'
    """

    test_bids = ['1240', '1241', '260', '223', '999']

    for bid in test_bids:
        result = extract_element_description(bid, state_text)
        print(f"\nBID: {bid}")
        if result:
            print(f"  Role: {result['role']}")
            print(f"  Label: {result['label']}")
            print(f"  Full: {result['full_description']}")
        else:
            print(f"  Not found")


def test_normalize_action():
    """Test action normalization."""
    print("\n\n=== Testing normalize_action ===")

    # Example state
    state_text = """
    [1240] combobox 'Sort by', expanded=False
    [1289] combobox 'Sort by', expanded=False
    [260] textbox 'Card Holder Name', required=True
    [256] textbox 'Card Holder Name', required=True
    [223] link 'My Account'
    """

    test_actions = [
        ("select_option('1240', 'Price')", state_text),
        ("select_option('1289', 'Price')", state_text),
        ("fill('260', 'switch card holder')", state_text),
        ("fill('256', 'switch card holder')", state_text),
        ("click('223')", state_text)
    ]

    for action, state in test_actions:
        result = normalize_action(action, state)
        print(f"\nRaw action: {result['raw_action']}")
        print(f"  Action type: {result['action_type']}")
        if result['element']:
            print(f"  Element role: {result['element']['role']}")
            print(f"  Element label: {result['element']['label']}")
        print(f"  Params: {result['params']}")
        print(f"  Normalized: {result['normalized_action']}")
        print(f"  Semantic: {result['semantic_action']}")


def test_calculate_action_similarity():
    """Test action similarity calculation."""
    print("\n\n=== Testing calculate_action_similarity ===")

    # Example state
    state_text1 = """
    [1240] combobox 'Sort by', expanded=False
    [260] textbox 'Card Holder Name', required=True
    """

    state_text2 = """
    [1289] combobox 'Sort by', expanded=False
    [256] textbox 'Card Holder Name', required=True
    [789] combobox 'Items per page', expanded=False
    """

    test_pairs = [
        # Same action, different BID (should be 1.0)
        ("select_option('1240', 'Price')", state_text1,
         "select_option('1289', 'Price')", state_text2),

        # Same action type and element, different params (should be ~0.6)
        ("select_option('1240', 'Price')", state_text1,
         "select_option('1289', 'Name')", state_text2),

        # Same element, different action type (should be 0.0)
        ("click('260')", state_text1,
         "fill('256', 'John Doe')", state_text2),

        # Different elements (should be 0.0)
        ("select_option('1240', 'Price')", state_text1,
         "select_option('789', 'Price')", state_text2),
    ]

    for action1, state1, action2, state2 in test_pairs:
        norm1 = normalize_action(action1, state1)
        norm2 = normalize_action(action2, state2)
        similarity = calculate_action_similarity(norm1, norm2)

        print(f"\n--- Comparison ---")
        print(f"Action 1: {action1}")
        print(f"  Normalized: {norm1['normalized_action']}")
        print(f"Action 2: {action2}")
        print(f"  Normalized: {norm2['normalized_action']}")
        print(f"Similarity: {similarity:.4f}")


def main():
    """Run all tests."""
    print("="*80)
    print("ACTION NORMALIZATION TEST SUITE")
    print("="*80)

    test_parse_action()
    test_extract_element_description()
    test_normalize_action()
    test_calculate_action_similarity()

    print("\n\n" + "="*80)
    print("ALL TESTS COMPLETED")
    print("="*80)


if __name__ == "__main__":
    main()
python test_webarena_lite.py --tasks 0 --model openai/gpt-4o --max_steps 10 --disable_memory --workers 1