#!/usr/bin/env python3
"""
Caption Block 主系统接口
时间序列模式索引和检索系统的命令行接口
"""

import argparse
import sys
import time
from typing import Dict, List, Any
from data_loader import DataLoader
from index_builder import IndexBuilder
from smart_query_parser import SmartQueryParser
from search_engine import SearchEngine


class CaptionBlockSystem:
    """Caption Block 系统主类"""
    
    def __init__(self, data_dir: str = "output", index_dir: str = "index_cache"):
        self.data_dir = data_dir
        self.index_dir = index_dir
        
        # 初始化各个组件
        self.data_loader = DataLoader(data_dir)
        self.index_builder = IndexBuilder(index_dir)
        self.query_parser = SmartQueryParser()
        self.search_engine = None
        self.records = []
    
    def build_index(self, force_rebuild: bool = False) -> None:
        """构建完整索引"""
        if not force_rebuild:
            # 尝试加载现有索引
            success, records = self.index_builder.load_index()
            if success:
                self.records = records
                self._initialize_search_engine()
                return
        
        print("Building index system...")
        
        # 加载数据
        self.records = self.data_loader.load_csv_data()
        
        # 转换为JSON
        self.data_loader.convert_to_json(self.records)
        
        # 构建索引
        self.index_builder.build_pattern_blocks(self.records)
        self.index_builder.build_time_range_index(self.records)
        
        # 保存索引
        self.index_builder.save_index(self.records)
        
        # 初始化搜索引擎
        self._initialize_search_engine()
        
        print("Index building completed!")
    
    def _initialize_search_engine(self) -> None:
        """初始化搜索引擎"""
        pattern_blocks = self.index_builder.get_pattern_blocks()
        time_range_index = self.index_builder.get_time_range_index()
        
        self.search_engine = SearchEngine(
            records=self.records,
            pattern_blocks=pattern_blocks,
            time_range_index=time_range_index
        )
    
    def search(self, query: str) -> List[Dict[str, Any]]:
        """执行综合搜索"""
        if not self.search_engine:
            raise RuntimeError("Search engine not initialized. Please call build_index() first.")
        
        # 解析查询
        time_range, patterns = self.query_parser.parse_query(query)
        
        # 执行搜索
        return self.search_engine.search(time_range=time_range, patterns=patterns)
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        if not self.search_engine:
            return {"error": "Search engine not initialized"}
        
        pattern_stats = self.search_engine.get_pattern_statistics()
        time_coverage = self.search_engine.get_time_range_coverage()
        
        return {
            'total_records': len(self.records),
            'total_patterns': len(pattern_stats),
            'total_time_ranges': len(self.index_builder.get_time_range_index()),
            'pattern_distribution': pattern_stats,
            'time_coverage': {
                'start': time_coverage[0],
                'end': time_coverage[1]
            }
        }


def format_search_results(results: List[Dict[str, Any]], max_results: int = 10) -> None:
    """格式化并打印搜索结果"""
    if not results:
        print("No matching records found.")
        return
    
    print(f"\nSearch Results: {len(results)} records found")
    print("=" * 60)
    
    for i, result in enumerate(results[:max_results], 1):
        print(f"\nRecord {i}:")
        print(f"  Time Range: {result['range']}")
        print(f"  Pattern: {result['pattern']}")
        print(f"  Description: {result['caption']}")
    
    if len(results) > max_results:
        print(f"\n... and {len(results) - max_results} more records")


def run_interactive_mode(system: CaptionBlockSystem) -> None:
    """运行交互式查询模式"""
    print("\nInteractive Query Mode (type 'quit' to exit)")
    print("Example queries:")
    print("  - 从时间范围[44000, 46000]中查找异常（level shift）")
    print("  - upward trend")
    print("  - 时间范围[45000, 46000]")
    print("-" * 60)
    
    while True:
        try:
            query = input("\nEnter your query: ").strip()
            
            if query.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
                
            if not query:
                continue
            
            start_time = time.time()
            results = system.search(query)
            end_time = time.time()
            
            format_search_results(results, max_results=5)
            print(f"\nQuery time: {(end_time - start_time)*1000:.2f} ms")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error during search: {e}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description='Caption Block - Temporal Pattern Index and Retrieval System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python caption_block.py -i "从时间范围[44000, 46000]中查找异常（level shift）"
  python caption_block.py --interactive
  python caption_block.py --stats
  python caption_block.py --rebuild
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        help='Input query string for search'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show system statistics'
    )
    
    parser.add_argument(
        '--rebuild',
        action='store_true',
        help='Force rebuild index'
    )
    
    parser.add_argument(
        '--max-results',
        type=int,
        default=10,
        help='Maximum number of results to display (default: 10)'
    )
    
    parser.add_argument(
        '--data-dir',
        type=str,
        default='output',
        help='Data directory path (default: output)'
    )
    
    parser.add_argument(
        '--index-dir',
        type=str,
        default='index_cache',
        help='Index cache directory path (default: index_cache)'
    )
    
    args = parser.parse_args()
    
    # 初始化系统
    print("Initializing Caption Block System...")
    system = CaptionBlockSystem(data_dir=args.data_dir, index_dir=args.index_dir)
    
    # 构建索引
    system.build_index(force_rebuild=args.rebuild)
    
    # 根据参数执行不同操作
    if args.stats:
        # 显示统计信息
        stats = system.get_statistics()
        print("\nSystem Statistics:")
        print("=" * 40)
        print(f"Total Records: {stats['total_records']}")
        print(f"Pattern Types: {stats['total_patterns']}")
        print(f"Time Ranges: {stats['total_time_ranges']}")
        print(f"Time Coverage: {stats['time_coverage']['start']} - {stats['time_coverage']['end']}")
        
        print("\nPattern Distribution:")
        for pattern, info in stats['pattern_distribution'].items():
            percentage = (info['records'] / stats['total_records']) * 100
            print(f"  {pattern}: {info['records']} records ({percentage:.1f}%)")
    
    elif args.input:
        # 执行单次查询
        print(f"\nExecuting query: '{args.input}'")
        
        start_time = time.time()
        results = system.search(args.input)
        end_time = time.time()
        
        format_search_results(results, max_results=args.max_results)
        print(f"\nQuery execution time: {(end_time - start_time)*1000:.2f} ms")
    
    elif args.interactive:
        # 交互式模式
        run_interactive_mode(system)
    
    else:
        # 默认显示帮助信息
        parser.print_help()
        
        # 显示简单的系统信息
        stats = system.get_statistics()
        print(f"\nSystem ready with {stats['total_records']} records loaded.")
        print("Use --help for usage information.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)