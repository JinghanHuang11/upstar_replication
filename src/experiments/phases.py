"""
Implementation of all experiment phases

Phase 0: Data Preprocessing
Phase 1: Baseline Training
Phase 2: Item Representation Learning
Phase 3: STB Calculation
Phase 4: UPSTAR Training
Phase 5: Evaluation and Comparison
"""

import sys
import logging
import pickle
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Any
import subprocess
import json

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.experiments.phase_base import ExperimentPhase

logger = logging.getLogger(__name__)


class Phase0_Preprocess(ExperimentPhase):
    """Phase 0: Data preprocessing"""

    @property
    def phase_name(self) -> str:
        return "phase0_preprocess"

    @property
    def phase_number(self) -> int:
        return 0

    def check_dependencies(self) -> bool:
        """Check if raw data exists"""
        raw_dir = Path(self.config['dataset']['data_dir'])
        dataset_name = self.config['dataset']['name']

        # Check for common data file names
        possible_files = [
            raw_dir / f"{dataset_name}.csv",
            raw_dir / "ta_feng.csv",
            raw_dir / "data.csv"
        ]

        return any(f.exists() for f in possible_files)

    def run(self) -> Dict[str, Any]:
        """Run preprocessing"""
        logger.info("Running data preprocessing...")

        # Import here to avoid circular imports
        from src.data.preprocess import get_preprocessor

        # Create preprocessor (uses TafengPreprocessor for tafeng dataset)
        preprocessor = get_preprocessor(self.config)
        df = preprocessor.run()

        # Build sequences
        from src.data.build_sequences import SequenceBuilder

        builder = SequenceBuilder(self.config)
        builder.run(df)

        # Load counts from saved files
        processed_dir = Path(self.config['dataset']['processed_dir'])
        with open(processed_dir / 'train_sequences.pkl', 'rb') as f:
            train_seqs = pickle.load(f)
        with open(processed_dir / 'val_sequences.pkl', 'rb') as f:
            val_seqs = pickle.load(f)
        with open(processed_dir / 'test_sequences.pkl', 'rb') as f:
            test_seqs = pickle.load(f)

        results = {
            'num_users': preprocessor.num_users,
            'num_items': preprocessor.num_items,
            'num_interactions': preprocessor.num_interactions,
            'train_sequences': len(train_seqs),
            'val_sequences': len(val_seqs),
            'test_sequences': len(test_seqs)
        }

        logger.info(f"Preprocessing complete: {results}")
        return results


class Phase1_Baseline(ExperimentPhase):
    """Phase 1: Baseline training"""

    @property
    def phase_name(self) -> str:
        return "phase1_baseline"

    @property
    def phase_number(self) -> int:
        return 1

    def check_dependencies(self) -> bool:
        """Check if preprocessed data exists"""
        processed_dir = Path(self.config['dataset']['processed_dir'])
        metadata_file = processed_dir / 'metadata.pkl'
        return metadata_file.exists()

    def run(self) -> Dict[str, Any]:
        """Run baseline training"""
        logger.info("Running baseline training...")

        # Import training module
        from src.training.train_baseline import main as train_main

        # Create temporary args
        import argparse
        args = argparse.Namespace(
            config=self.config.get('baseline_config', 'configs/tafeng_baseline.yaml')
        )

        # Run training
        metrics = train_main(args)

        results = {
            'ndcg@10': float(metrics.get('ndcg_10', 0.0)),
            'hr@10': float(metrics.get('hr_10', 0.0)),
            'mrr@10': float(metrics.get('mrr_10', 0.0)),
            'best_epoch': int(metrics.get('best_epoch', 0)),
            'model_path': str(self.checkpoint_dir / "best_model.pt")
        }

        logger.info(f"Baseline training complete: {results}")
        return results


class Phase2_ItemRepr(ExperimentPhase):
    """Phase 2: Item representation learning"""

    @property
    def phase_name(self) -> str:
        return "phase2_item_repr"

    @property
    def phase_number(self) -> int:
        return 2

    def check_dependencies(self) -> bool:
        """Check if preprocessed data and graphs exist"""
        processed_dir = Path(self.config['dataset']['processed_dir'])
        metadata_file = processed_dir / 'metadata.pkl'
        return metadata_file.exists()

    def run(self) -> Dict[str, Any]:
        """Run item representation learning"""
        logger.info("Running item representation learning...")

        from src.training.train_item_repr import main as train_main

        import argparse
        args = argparse.Namespace(
            config=self.config.get('item_repr_config', 'configs/item_repr.yaml')
        )

        # Run training
        metrics = train_main(args)

        results = {
            'embedding_dim': self.config['model'].get('item_embed_dim', self.config['model'].get('embed_dim', 128)),
            'num_items': metrics.get('num_items', 0),
            'embedding_path': str(self.checkpoint_dir / "item_embeddings.pt")
        }

        logger.info(f"Item representation learning complete: {results}")
        return results


class Phase3_STB(ExperimentPhase):
    """Phase 3: STB calculation"""

    @property
    def phase_name(self) -> str:
        return "phase3_stb"

    @property
    def phase_number(self) -> int:
        return 3

    def check_dependencies(self) -> bool:
        """Check if item representations exist"""
        embedding_path = Path(self.config['stb'].get('item_embeddings_path',
                                                     'outputs/phase2_item_repr/checkpoints/item_embeddings.pt'))
        return embedding_path.exists()

    def run(self) -> Dict[str, Any]:
        """Run STB calculation"""
        logger.info("Running STB calculation...")

        from src.training.train_stb import main as train_main

        import argparse
        args = argparse.Namespace(
            config=self.config.get('stb_config', 'configs/stb.yaml')
        )

        # Run STB calculation
        stb_results = train_main(args)

        # Load motivation labels
        motivation_labels_path = Path(self.config['stb']['motivation_labels_path'])
        motivation_labels = np.load(motivation_labels_path)

        # Count labels
        unique, counts = np.unique(motivation_labels, return_counts=True)
        label_dist = dict(zip(unique.tolist(), counts.tolist()))

        results = {
            'num_items': len(motivation_labels),
            'stable_count': label_dist.get(1, 0),
            'exploratory_count': label_dist.get(0, 0),
            'other_count': label_dist.get(2, 0),
            'stb_scores_path': str(self.config['stb']['stb_scores_path']),
            'motivation_labels_path': str(self.config['stb']['motivation_labels_path'])
        }

        logger.info(f"STB calculation complete: {results}")
        return results


class Phase4_UPSTAR(ExperimentPhase):
    """Phase 4: UPSTAR training"""

    @property
    def phase_name(self) -> str:
        return "phase4_upstar"

    @property
    def phase_number(self) -> int:
        return 4

    def check_dependencies(self) -> bool:
        """Check if STB results exist"""
        motivation_labels_path = Path(self.config['stb']['motivation_labels_path'])
        item_embeddings_path = Path(self.config['stb'].get('item_embeddings_path',
                                                           'outputs/phase2_item_repr/checkpoints/item_embeddings.pt'))
        return motivation_labels_path.exists() and item_embeddings_path.exists()

    def run(self) -> Dict[str, Any]:
        """Run UPSTAR training"""
        logger.info("Running UPSTAR training (4 stages)...")

        from src.training.train_upstar import main as train_main

        import argparse
        args = argparse.Namespace(
            config=self.config.get('upstar_config', 'configs/tafeng_upstar.yaml')
        )

        # Run training
        metrics = train_main(args)

        results = {
            'ndcg@10': float(metrics.get('ndcg_10', 0.0)),
            'hr@10': float(metrics.get('hr_10', 0.0)),
            'mrr@10': float(metrics.get('mrr_10', 0.0)),
            'model_path': str(self.checkpoint_dir / "model_after_stage4.pt")
        }

        logger.info(f"UPSTAR training complete: {results}")
        return results


class Phase5_Evaluate(ExperimentPhase):
    """Phase 5: Evaluation and comparison"""

    @property
    def phase_name(self) -> str:
        return "phase5_evaluate"

    @property
    def phase_number(self) -> int:
        return 5

    def check_dependencies(self) -> bool:
        """Check if both models are trained"""
        baseline_model = Path(self.config['baseline'].get('model_path',
                                                          'outputs/phase1_baseline/checkpoints/best_model.pt'))
        upstar_model = Path(self.config['upstar'].get('model_path',
                                                      'outputs/phase4_upstar/checkpoints/model_after_stage4.pt'))
        return baseline_model.exists() and upstar_model.exists()

    def run(self) -> Dict[str, Any]:
        """Run evaluation and comparison"""
        logger.info("Running evaluation and comparison...")

        # Load baseline results
        baseline_results = self.load_baseline_results()

        # Load UPSTAR results
        upstar_results = self.load_upstar_results()

        # Calculate improvement
        comparison = self.calculate_improvement(baseline_results, upstar_results)

        results = {
            'baseline': baseline_results,
            'upstar': upstar_results,
            'comparison': comparison
        }

        # Save detailed comparison
        self.save_comparison_table(results)

        logger.info(f"Evaluation complete: {comparison}")
        return results

    def load_baseline_results(self) -> Dict[str, Any]:
        """Load baseline evaluation results"""
        # Try to load from phase 1 results
        phase1_results = Phase1_Baseline(self.config, self.output_dir).load_results()
        if phase1_results:
            return phase1_results

        # Fallback to default values
        return {
            'ndcg@10': 0.0,
            'hr@10': 0.0,
            'mrr@10': 0.0
        }

    def load_upstar_results(self) -> Dict[str, Any]:
        """Load UPSTAR evaluation results"""
        # Try to load from phase 4 results
        phase4_results = Phase4_UPSTAR(self.config, self.output_dir).load_results()
        if phase4_results:
            return phase4_results

        # Fallback to default values
        return {
            'ndcg@10': 0.0,
            'hr@10': 0.0,
            'mrr@10': 0.0
        }

    def calculate_improvement(self, baseline: Dict, upstar: Dict) -> Dict[str, Any]:
        """Calculate percentage improvement"""
        comparison = {}

        for metric in ['ndcg@10', 'hr@10', 'mrr@10']:
            baseline_val = baseline.get(metric, 0.0)
            upstar_val = upstar.get(metric, 0.0)

            if baseline_val > 0:
                improvement = ((upstar_val - baseline_val) / baseline_val) * 100
            else:
                improvement = 0.0

            comparison[metric] = {
                'baseline': baseline_val,
                'upstar': upstar_val,
                'improvement_pct': improvement,
                'absolute_diff': upstar_val - baseline_val
            }

        return comparison

    def save_comparison_table(self, results: Dict[str, Any]):
        """Save comparison table as text and JSON"""
        comparison = results['comparison']

        # Create table
        table = "\n" + "="*80 + "\n"
        table += "UPSTAR vs Baseline - Performance Comparison\n"
        table += "="*80 + "\n\n"
        table += f"{'Metric':<15} {'Baseline':<12} {'UPSTAR':<12} {'Improvement':<12}\n"
        table += "-"*80 + "\n"

        for metric_name, values in comparison.items():
            metric_display = metric_name.replace('@', '@').upper()
            baseline = values['baseline']
            upstar = values['upstar']
            improvement = values['improvement_pct']

            table += f"{metric_display:<15} {baseline:<12.4f} {upstar:<12.4f} {improvement:+.2f}%\n"

        table += "="*80 + "\n"

        # Save as text
        table_file = self.results_dir / "comparison_table.txt"
        with open(table_file, 'w') as f:
            f.write(table)

        # Save as JSON
        json_file = self.results_dir / "comparison.json"
        with open(json_file, 'w') as f:
            json.dump(comparison, f, indent=2)

        logger.info(f"Comparison table saved to {table_file}")