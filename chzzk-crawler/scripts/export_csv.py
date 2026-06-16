import sys
import os
import pandas as pd
import argparse
from sqlalchemy import create_engine

# Add parent dir to path to import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.settings import get_settings

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--table', required=True, help='Table name (chat_messages_raw or minute_features)')
    parser.add_argument('--out', help='Output filename (auto-generated if not provided)')
    parser.add_argument('--run_id', type=int, help='Optional run_id filter')
    parser.add_argument('--excel', action='store_true', help='Export to Excel instead of CSV')
    args = parser.parse_args()

    settings = get_settings()
    engine = create_engine(settings.sqlalchemy_url)
    
    query = f"SELECT * FROM {args.table}"
    if args.run_id:
        query += f" WHERE run_id = {args.run_id}"
        
    out_file = args.out
    if not out_file:
        ext = 'xlsx' if args.excel else 'csv'
        out_file = f"CHZZK_Run_{args.run_id or 'all'}_{args.table}_{pd.Timestamp.now().strftime('%Y%H%M%S')}.{ext}"

    print(f'Exporting {args.table} to {out_file} (run_id={args.run_id})...')
    try:
        df = pd.read_sql(query, engine)
        if args.excel:
            df.to_excel(out_file, index=False)
        else:
            df.to_csv(out_file, index=False, encoding='utf-8-sig')
        print(f'Success! {len(df)} rows exported.')
    except Exception as e:
        print(f'Error: {e}')

if __name__ == '__main__':
    main()
