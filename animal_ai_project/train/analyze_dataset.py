"""
Dataset Analysis, Quality Assessment & Improvement Strategy
===========================================================

This script provides complete dataset diagnostics and improvement recommendations.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image
from collections import defaultdict
import hashlib


class DatasetAnalyzer:
    """Complete dataset quality assessment"""
    
    def __init__(self, dataset_dir: str):
        """
        Args:
            dataset_dir: Root directory with class folders
                Example: datasets/dog/breed/
                    ├── chihuahua/
                    │   ├── img1.jpg
                    │   └── img2.jpg
                    ├── husky/
                    └── pug/
        """
        self.dataset_dir = dataset_dir
        self.class_stats = {}
        self.issues = []
        
    def analyze(self):
        """Run complete dataset analysis"""
        print("\n" + "="*70)
        print("DATASET QUALITY ANALYSIS")
        print("="*70)
        
        # Scan dataset
        self._scan_dataset()
        
        # Analyze class distribution
        self._analyze_class_distribution()
        
        # Check image quality
        self._check_image_quality()
        
        # Check for duplicates
        self._check_duplicates()
        
        # Generate recommendations
        self._generate_recommendations()
        
        # Print summary
        self._print_summary()
    
    def _scan_dataset(self):
        """Scan and count images per class"""
        print("\n📁 SCANNING DATASET...")
        
        for class_name in os.listdir(self.dataset_dir):
            class_path = os.path.join(self.dataset_dir, class_name)
            if not os.path.isdir(class_path):
                continue
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.webp'))]
            
            self.class_stats[class_name] = {
                'count': len(images),
                'images': images,
                'path': class_path,
                'issues': []
            }
        
        print(f"✅ Found {len(self.class_stats)} classes")
    
    def _analyze_class_distribution(self):
        """Analyze class balance"""
        print("\n⚖️  CLASS DISTRIBUTION ANALYSIS")
        print("-" * 70)
        
        counts = [stat['count'] for stat in self.class_stats.values()]
        total = sum(counts)
        
        print(f"\n📊 Class Distribution:")
        print(f"{'Class':<20} {'Count':<8} {'Percent':<10} {'Balance'}")
        print("-" * 70)
        
        max_count = max(counts)
        min_count = min(counts)
        
        for class_name, stat in sorted(self.class_stats.items()):
            count = stat['count']
            pct = (count / total * 100) if total > 0 else 0
            bar = "█" * (count // (max_count // 20) + 1)
            print(f"{class_name:<20} {count:<8} {pct:<10.1f} {bar}")
        
        print("-" * 70)
        print(f"Total images: {total}")
        
        # Imbalance analysis
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print(f"\n📈 Imbalance Metrics:")
        print(f"   Max count:  {max_count}")
        print(f"   Min count:  {min_count}")
        print(f"   Ratio:      {imbalance_ratio:.1f}x")
        print(f"   Avg/class:  {total/len(self.class_stats):.0f}")
        
        if imbalance_ratio > 3:
            msg = f"❌ SEVERE IMBALANCE ({imbalance_ratio:.1f}x) - Use class_weight in training"
            self.issues.append(msg)
            print(f"\n{msg}")
        elif imbalance_ratio > 1.5:
            msg = f"⚠️  Imbalance detected ({imbalance_ratio:.1f}x) - Consider data augmentation"
            self.issues.append(msg)
            print(f"\n{msg}")
        else:
            print(f"\n✅ Good balance ({imbalance_ratio:.1f}x)")
    
    def _check_image_quality(self):
        """Check image quality issues"""
        print("\n🔍 IMAGE QUALITY ANALYSIS")
        print("-" * 70)
        
        quality_issues = defaultdict(list)
        total_checked = 0
        
        for class_name, stat in self.class_stats.items():
            for image_file in stat['images']:
                image_path = os.path.join(stat['path'], image_file)
                total_checked += 1
                
                try:
                    img = Image.open(image_path)
                    
                    # Check size
                    if img.size[0] < 100 or img.size[1] < 100:
                        quality_issues['too_small'].append(
                            (class_name, image_file, img.size)
                        )
                    
                    # Check format
                    if img.mode != 'RGB':
                        quality_issues['not_rgb'].append(
                            (class_name, image_file, img.mode)
                        )
                    
                    # Check aspect ratio (too extreme)
                    aspect = img.size[0] / img.size[1]
                    if aspect > 3 or aspect < 0.33:
                        quality_issues['extreme_aspect'].append(
                            (class_name, image_file, f"{aspect:.2f}")
                        )
                    
                except Exception as e:
                    quality_issues['corrupted'].append(
                        (class_name, image_file, str(e))
                    )
        
        print(f"✅ Checked {total_checked} images")
        
        if not quality_issues:
            print("✅ No quality issues detected!")
        else:
            print(f"\n⚠️  Quality Issues Found:")
            for issue_type, items in quality_issues.items():
                print(f"\n   {issue_type.upper()} ({len(items)} images):")
                for class_name, img_file, detail in items[:3]:
                    print(f"      {class_name}/{img_file}: {detail}")
                if len(items) > 3:
                    print(f"      ... and {len(items)-3} more")
                
                self.issues.append(f"{issue_type}: {len(items)} images")
    
    def _check_duplicates(self):
        """Check for duplicate images"""
        print("\n🔎 DUPLICATE DETECTION")
        print("-" * 70)
        
        hashes = {}
        duplicates = []
        total = 0
        
        for class_name, stat in self.class_stats.items():
            for image_file in stat['images']:
                image_path = os.path.join(stat['path'], image_file)
                total += 1
                
                try:
                    # File hash
                    with open(image_path, 'rb') as f:
                        file_hash = hashlib.md5(f.read()).hexdigest()
                    
                    if file_hash in hashes:
                        duplicates.append((
                            f"{class_name}/{image_file}",
                            hashes[file_hash]
                        ))
                    else:
                        hashes[file_hash] = f"{class_name}/{image_file}"
                except:
                    pass
        
        print(f"✅ Checked {total} images for duplicates")
        
        if duplicates:
            print(f"\n⚠️  {len(duplicates)} duplicate files found:")
            for dup, original in duplicates[:5]:
                print(f"   {dup} → {original}")
            if len(duplicates) > 5:
                print(f"   ... and {len(duplicates)-5} more")
            self.issues.append(f"duplicates: {len(duplicates)} files")
        else:
            print("✅ No exact duplicates detected")
    
    def _generate_recommendations(self):
        """Generate improvement recommendations"""
        print("\n" + "="*70)
        print("RECOMMENDATIONS")
        print("="*70)
        
        total_images = sum(s['count'] for s in self.class_stats.values())
        avg_per_class = total_images / len(self.class_stats) if self.class_stats else 0
        
        recommendations = []
        
        # Dataset size recommendations
        if total_images < 300:
            recommendations.append("⚠️  Dataset too small (<300 total)")
            recommendations.append("   Action: Collect more images (target 500-1000)")
        elif total_images < 600:
            recommendations.append("⚠️  Dataset small (300-600 total)")
            recommendations.append("   Action: Increase to 1000+ images if possible")
        else:
            recommendations.append("✅ Dataset size adequate")
        
        # Per-class minimum
        min_per_class = min((s['count'] for s in self.class_stats.values()), default=0)
        if min_per_class < 30:
            recommendations.append(f"❌ Minimum per class is {min_per_class} (<30)")
            recommendations.append("   Action: Collect at least 50 images per class")
        elif min_per_class < 50:
            recommendations.append(f"⚠️  Minimum per class is {min_per_class} (<50)")
            recommendations.append("   Action: Target 100+ images per class")
        else:
            recommendations.append("✅ Sufficient samples per class")
        
        # Class balance
        counts = [s['count'] for s in self.class_stats.values()]
        imbalance = max(counts) / min(counts) if min(counts) > 0 else float('inf')
        if imbalance > 3:
            recommendations.append(f"❌ Severe imbalance ({imbalance:.1f}x)")
            recommendations.append("   Action: Use class_weight in training or collect more minority images")
        elif imbalance > 1.5:
            recommendations.append(f"⚠️  Moderate imbalance ({imbalance:.1f}x)")
            recommendations.append("   Action: Consider rebalancing via augmentation")
        
        # Number of classes
        num_classes = len(self.class_stats)
        if num_classes > 10:
            recommendations.append(f"⚠️  Many classes ({num_classes})")
            recommendations.append("   Action: Start with 3-5 classes for better accuracy")
        
        print("\n" + "\n".join(recommendations))
    
    def _print_summary(self):
        """Print final summary"""
        print("\n" + "="*70)
        print("SUMMARY & NEXT STEPS")
        print("="*70)
        
        if self.issues:
            print(f"\n⚠️  Found {len(self.issues)} issue(s):")
            for issue in self.issues:
                print(f"   • {issue}")
        else:
            print("\n✅ No major issues detected!")
        
        total = sum(s['count'] for s in self.class_stats.values())
        print(f"\n📊 Quick Stats:")
        print(f"   Classes: {len(self.class_stats)}")
        print(f"   Total images: {total}")
        print(f"   Avg/class: {total/len(self.class_stats):.0f}" if self.class_stats else "   Avg/class: 0")
        
        print(f"\n🎯 Optimization Steps:")
        print(f"   1. Fix any corrupted/low-quality images")
        print(f"   2. Ensure min 50 images per class (preferably 100+)")
        print(f"   3. Balance classes (target <2x imbalance ratio)")
        print(f"   4. Use data augmentation in training")
        print(f"   5. Start with 3-5 classes, add more later")
        print(f"   6. Monitor validation accuracy > 90%")


if __name__ == "__main__":
    # Example usage
    dataset_dir = "datasets/dog/breed"  # Change as needed
    
    analyzer = DatasetAnalyzer(dataset_dir)
    analyzer.analyze()
