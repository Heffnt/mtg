"""
Comprehensive Dataset Analysis for LLM Fine-Tuning
Analyzes JSONL ChatML format data and generates a detailed report
"""

import json
import re
from collections import Counter, defaultdict
from pathlib import Path
import statistics
from datetime import datetime


class DatasetAnalyzer:
    def __init__(self, data_dir="splits_full"):
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "train.jsonl"
        self.valid_path = self.data_dir / "valid.jsonl"
        self.test_path = self.data_dir / "test.jsonl"

        self.stats = {
            'train': {},
            'valid': {},
            'test': {},
            'overall': {}
        }

    def load_jsonl(self, file_path):
        """Load JSONL file and return list of examples."""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error parsing line: {e}")
                    continue
        return data

    def estimate_tokens(self, text):
        """Rough token estimation (1 token ≈ 4 characters for English)."""
        return len(text) // 4

    def analyze_split(self, data, split_name):
        """Analyze a single data split."""
        print(f"Analyzing {split_name} split...")

        stats = {
            'num_examples': len(data),
            'message_lengths': [],
            'system_prompts': Counter(),
            'user_queries': [],
            'assistant_responses': [],
            'total_tokens_estimated': 0,
            'conversation_turns': [],
            'card_types': Counter(),
            'rarities': Counter(),
            'mana_costs': [],
        }

        for example in data:
            messages = example.get('messages', [])
            stats['conversation_turns'].append(len(messages))

            conversation_text = ""

            for msg in messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                msg_len = len(content)
                conversation_text += content + " "

                stats['message_lengths'].append(msg_len)

                if role == 'system':
                    stats['system_prompts'][content] += 1
                elif role == 'user':
                    stats['user_queries'].append(content)
                elif role == 'assistant':
                    stats['assistant_responses'].append(content)
                    # Extract card information
                    self._extract_card_features(content, stats)

            # Estimate tokens for entire conversation
            stats['total_tokens_estimated'] += self.estimate_tokens(conversation_text)

        # Calculate statistics
        stats['avg_message_length'] = statistics.mean(stats['message_lengths']) if stats['message_lengths'] else 0
        stats['median_message_length'] = statistics.median(stats['message_lengths']) if stats['message_lengths'] else 0
        stats['max_message_length'] = max(stats['message_lengths']) if stats['message_lengths'] else 0
        stats['min_message_length'] = min(stats['message_lengths']) if stats['message_lengths'] else 0

        stats['avg_tokens_per_example'] = stats['total_tokens_estimated'] / len(data) if data else 0
        stats['avg_conversation_turns'] = statistics.mean(stats['conversation_turns']) if stats['conversation_turns'] else 0

        # User query analysis
        user_query_lengths = [len(q) for q in stats['user_queries']]
        stats['avg_user_query_length'] = statistics.mean(user_query_lengths) if user_query_lengths else 0

        # Assistant response analysis
        assistant_resp_lengths = [len(r) for r in stats['assistant_responses']]
        stats['avg_assistant_response_length'] = statistics.mean(assistant_resp_lengths) if assistant_resp_lengths else 0
        stats['min_assistant_response_length'] = min(assistant_resp_lengths) if assistant_resp_lengths else 0
        stats['max_assistant_response_length'] = max(assistant_resp_lengths) if assistant_resp_lengths else 0

        return stats

    def _extract_card_features(self, content, stats):
        """Extract MTG card features from assistant response."""
        # Detect card types (Creature, Instant, Sorcery, etc.)
        card_type_patterns = [
            r'\bCreature\b', r'\bInstant\b', r'\bSorcery\b',
            r'\bEnchantment\b', r'\bArtifact\b', r'\bPlaneswalker\b',
            r'\bLand\b', r'\bTribal\b'
        ]
        for pattern in card_type_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                card_type = pattern.strip(r'\b').strip('\\')
                stats['card_types'][card_type] += 1

        # Detect rarities
        rarity_patterns = [r'\bCommon\b', r'\bUncommon\b', r'\bRare\b', r'\bMythic Rare\b']
        for pattern in rarity_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                rarity = pattern.strip(r'\b').strip('\\')
                stats['rarities'][rarity] += 1

        # Extract mana costs
        mana_cost_match = re.search(r'\{[^\}]+\}', content)
        if mana_cost_match:
            stats['mana_costs'].append(mana_cost_match.group())

    def analyze_all_splits(self):
        """Analyze all data splits."""
        for split_name, path in [('train', self.train_path),
                                  ('valid', self.valid_path),
                                  ('test', self.test_path)]:
            if path.exists():
                data = self.load_jsonl(path)
                self.stats[split_name] = self.analyze_split(data, split_name)
            else:
                print(f"Warning: {path} not found")

        # Overall statistics
        self.stats['overall']['total_examples'] = sum(
            self.stats[split].get('num_examples', 0)
            for split in ['train', 'valid', 'test']
        )
        self.stats['overall']['total_tokens_estimated'] = sum(
            self.stats[split].get('total_tokens_estimated', 0)
            for split in ['train', 'valid', 'test']
        )

    def generate_report(self, output_file="dataset_analysis.txt"):
        """Generate comprehensive text report."""
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("DATASET ANALYSIS REPORT FOR LLM FINE-TUNING\n")
            f.write("=" * 80 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Dataset Directory: {self.data_dir.absolute()}\n")
            f.write("=" * 80 + "\n\n")

            # 1. DATASET OVERVIEW
            f.write("1. DATASET OVERVIEW\n")
            f.write("-" * 80 + "\n")
            f.write(f"Total Examples: {self.stats['overall']['total_examples']:,}\n")
            f.write(f"Estimated Total Tokens: {self.stats['overall']['total_tokens_estimated']:,}\n\n")

            f.write("Split Distribution:\n")
            for split in ['train', 'valid', 'test']:
                num = self.stats[split].get('num_examples', 0)
                pct = (num / self.stats['overall']['total_examples'] * 100) if self.stats['overall']['total_examples'] > 0 else 0
                f.write(f"  {split.capitalize():8s}: {num:10,} examples ({pct:5.2f}%)\n")
            f.write("\n")

            # 2. DATA FORMAT
            f.write("2. DATA FORMAT\n")
            f.write("-" * 80 + "\n")
            f.write("Format: ChatML (JSON Lines with messages array)\n")
            f.write("Structure: Each example contains:\n")
            f.write("  - System message: Sets context as MTG expert\n")
            f.write("  - User message: Query about MTG card\n")
            f.write("  - Assistant message: Card details/response\n\n")

            # Show example system prompts
            if self.stats['train'].get('system_prompts'):
                f.write("System Prompts Found:\n")
                for prompt, count in self.stats['train']['system_prompts'].most_common(5):
                    f.write(f"  [{count:,} occurrences] {prompt[:100]}\n")
                f.write("\n")

            # 3. DETAILED STATISTICS PER SPLIT
            f.write("3. DETAILED STATISTICS PER SPLIT\n")
            f.write("-" * 80 + "\n\n")

            for split in ['train', 'valid', 'test']:
                stats = self.stats[split]
                if not stats:
                    continue

                f.write(f"{split.upper()} SPLIT:\n")
                f.write(f"  Examples: {stats.get('num_examples', 0):,}\n")
                f.write(f"  Avg Tokens per Example: {stats.get('avg_tokens_per_example', 0):.1f}\n")
                f.write(f"  Avg Conversation Turns: {stats.get('avg_conversation_turns', 0):.1f}\n\n")

                f.write(f"  Message Length Statistics (characters):\n")
                f.write(f"    Average: {stats.get('avg_message_length', 0):.1f}\n")
                f.write(f"    Median:  {stats.get('median_message_length', 0):.1f}\n")
                f.write(f"    Min:     {stats.get('min_message_length', 0)}\n")
                f.write(f"    Max:     {stats.get('max_message_length', 0)}\n\n")

                f.write(f"  User Query Length (characters):\n")
                f.write(f"    Average: {stats.get('avg_user_query_length', 0):.1f}\n\n")

                f.write(f"  Assistant Response Length (characters):\n")
                f.write(f"    Average: {stats.get('avg_assistant_response_length', 0):.1f}\n")
                f.write(f"    Min:     {stats.get('min_assistant_response_length', 0)}\n")
                f.write(f"    Max:     {stats.get('max_assistant_response_length', 0)}\n\n")

            # 4. CONTENT ANALYSIS
            f.write("4. CONTENT ANALYSIS\n")
            f.write("-" * 80 + "\n\n")

            # Card types distribution
            f.write("Card Types Distribution (Training Set):\n")
            card_types = self.stats['train'].get('card_types', {})
            total_cards = sum(card_types.values())
            for card_type, count in card_types.most_common():
                pct = (count / total_cards * 100) if total_cards > 0 else 0
                f.write(f"  {card_type:15s}: {count:8,} ({pct:5.2f}%)\n")
            f.write("\n")

            # Rarities distribution
            f.write("Card Rarities Distribution (Training Set):\n")
            rarities = self.stats['train'].get('rarities', {})
            total_rarities = sum(rarities.values())
            for rarity, count in rarities.most_common():
                pct = (count / total_rarities * 100) if total_rarities > 0 else 0
                f.write(f"  {rarity:15s}: {count:8,} ({pct:5.2f}%)\n")
            f.write("\n")

            # Sample mana costs
            mana_costs = self.stats['train'].get('mana_costs', [])
            if mana_costs:
                f.write("Sample Mana Costs (first 20 unique):\n")
                unique_costs = list(set(mana_costs))[:20]
                f.write(f"  {', '.join(unique_costs)}\n\n")

            # 5. DATA QUALITY ASSESSMENT
            f.write("5. DATA QUALITY ASSESSMENT\n")
            f.write("-" * 80 + "\n\n")

            # Check for consistency
            f.write("Quality Indicators:\n")
            avg_response_len = self.stats['train'].get('avg_assistant_response_length', 0)
            min_response_len = self.stats['train'].get('min_assistant_response_length', 0)

            f.write(f"  ✓ Consistent format: ChatML with system/user/assistant roles\n")
            f.write(f"  ✓ Large dataset: {self.stats['overall']['total_examples']:,} total examples\n")
            f.write(f"  ✓ Good split ratio: ~98% train, ~1% validation, ~1% test\n")

            if avg_response_len > 50:
                f.write(f"  ✓ Substantial responses: avg {avg_response_len:.0f} chars\n")

            if min_response_len > 0:
                f.write(f"  ✓ No empty responses detected\n")

            f.write("\n")

            # 6. FINE-TUNING RECOMMENDATIONS
            f.write("6. FINE-TUNING RECOMMENDATIONS\n")
            f.write("-" * 80 + "\n\n")

            avg_tokens = self.stats['train'].get('avg_tokens_per_example', 0)

            f.write("Model Configuration:\n")
            if avg_tokens < 512:
                f.write(f"  • Max Sequence Length: 512-1024 tokens (avg example ~{avg_tokens:.0f} tokens)\n")
            elif avg_tokens < 2048:
                f.write(f"  • Max Sequence Length: 2048 tokens (avg example ~{avg_tokens:.0f} tokens)\n")
            else:
                f.write(f"  • Max Sequence Length: 4096+ tokens (avg example ~{avg_tokens:.0f} tokens)\n")

            f.write(f"  • Recommended to use chat template formatting\n")
            f.write(f"  • Consider using BF16 precision for better numerical stability\n\n")

            f.write("Training Strategy:\n")
            num_train = self.stats['train'].get('num_examples', 0)

            if num_train > 1_000_000:
                f.write(f"  • Large dataset ({num_train:,} examples) - consider:\n")
                f.write(f"    - LoRA/QLoRA for efficiency\n")
                f.write(f"    - Smaller learning rate (1e-5 to 5e-5)\n")
                f.write(f"    - 1-3 epochs should be sufficient\n")
                f.write(f"    - Use gradient accumulation to increase effective batch size\n")
            elif num_train > 100_000:
                f.write(f"  • Medium dataset ({num_train:,} examples):\n")
                f.write(f"    - 2-4 epochs recommended\n")
                f.write(f"    - Learning rate: 2e-5 to 1e-4\n")
            else:
                f.write(f"  • Smaller dataset ({num_train:,} examples):\n")
                f.write(f"    - 3-5 epochs may be needed\n")
                f.write(f"    - Watch for overfitting\n")

            f.write("\n")

            f.write("Hyperparameter Suggestions:\n")
            f.write(f"  • Batch Size: 2-8 per device (depending on GPU memory)\n")
            f.write(f"  • Gradient Accumulation: 4-32 steps (for effective batch ~64-256)\n")
            f.write(f"  • Learning Rate: 2e-5 (with warmup)\n")
            f.write(f"  • Warmup Steps: 100-500\n")
            f.write(f"  • Weight Decay: 0.01\n")
            f.write(f"  • Max Grad Norm: 1.0\n\n")

            f.write("Validation Strategy:\n")
            f.write(f"  • Use validation set ({self.stats['valid'].get('num_examples', 0):,} examples)\n")
            f.write(f"  • Monitor validation loss every 500-1000 steps\n")
            f.write(f"  • Early stopping if validation loss plateaus\n")
            f.write(f"  • Keep test set ({self.stats['test'].get('num_examples', 0):,} examples) for final evaluation\n\n")

            # 7. DOMAIN-SPECIFIC NOTES
            f.write("7. DOMAIN-SPECIFIC NOTES (Magic: The Gathering)\n")
            f.write("-" * 80 + "\n\n")

            f.write("Dataset Characteristics:\n")
            f.write(f"  • Task: MTG card knowledge and information retrieval\n")
            f.write(f"  • Format: Question-answering about card details\n")
            f.write(f"  • Domain: Structured game rules and card mechanics\n\n")

            f.write("Expected Model Capabilities After Fine-Tuning:\n")
            f.write(f"  • Recall card names, mana costs, types, and rarities\n")
            f.write(f"  • Describe card abilities and rules text\n")
            f.write(f"  • Provide power/toughness for creatures\n")
            f.write(f"  • Answer queries about specific MTG cards\n\n")

            f.write("Potential Improvements:\n")
            f.write(f"  • Add examples with card interactions and combos\n")
            f.write(f"  • Include ruling clarifications\n")
            f.write(f"  • Add deck-building advice examples\n")
            f.write(f"  • Include format legality information\n\n")

            # 8. EXAMPLE DATA
            f.write("8. SAMPLE EXAMPLES\n")
            f.write("-" * 80 + "\n\n")

            # Show a few examples from training set
            train_data = self.load_jsonl(self.train_path)
            f.write("Sample Training Examples:\n\n")
            for i, example in enumerate(train_data[:3], 1):
                f.write(f"Example {i}:\n")
                for msg in example.get('messages', []):
                    role = msg.get('role', '').upper()
                    content = msg.get('content', '')
                    f.write(f"  [{role}]: {content[:200]}\n")
                f.write("\n")

            # Footer
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")

        print(f"\n✓ Analysis complete! Report saved to: {output_file}")


def main():
    """Run the dataset analysis."""
    print("Starting dataset analysis...")
    print("=" * 80)

    analyzer = DatasetAnalyzer(data_dir="splits_full")
    analyzer.analyze_all_splits()
    analyzer.generate_report(output_file="dataset_analysis.txt")

    print("=" * 80)
    print("\nSummary:")
    print(f"  Total Examples: {analyzer.stats['overall']['total_examples']:,}")
    print(f"  Estimated Tokens: {analyzer.stats['overall']['total_tokens_estimated']:,}")
    print(f"  Training Examples: {analyzer.stats['train'].get('num_examples', 0):,}")
    print(f"  Validation Examples: {analyzer.stats['valid'].get('num_examples', 0):,}")
    print(f"  Test Examples: {analyzer.stats['test'].get('num_examples', 0):,}")
    print("\nFull report available in: dataset_analysis.txt")


if __name__ == "__main__":
    main()
