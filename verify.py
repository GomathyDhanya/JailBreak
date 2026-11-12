
from collections import Counter
from pathlib import Path
import pandas as pd
from config import DATA, REFERENCE
from data_loading import load_issues,load_training_examples,load_cached_task1,attachment_path
from data_creation import issue_records,attachment,INJECTION,build_training_examples
from tools import extract,mass_mention

def main():
    issues=load_issues()
    training=load_training_examples()
    assert issues==issue_records()
    assert training==build_training_examples()
    for issue in issues:
        for condition in ('clean','injected'):
            expected=attachment(issue)+(INJECTION if condition=='injected' else '')
            assert attachment_path(issue['id'],condition).read_text(encoding='utf-8')==expected
    assert sum(c['scenario']=='injected' for c in training)==48
    assert sum(c['scenario']=='clean' for c in training)==48
    assert sum(c['scenario']=='authorized' for c in training)==48
    baseline,tests,gcg=load_cached_task1()
    assert len(baseline)==63 and len(tests)==50 and len(gcg['suffix_ids'])==20
    assert len(gcg['settings']['train_ids'])==25
    base=pd.read_csv(REFERENCE/'task2_base_results.csv')
    assert len(base)==300
    
    summary=base.groupby(['defense','condition'])['success'].sum()
    print('Dataset verified: 50 issues, 100 Markdown files, 144 training cases')
    print('Cached Task 1: 63 screened, 50 held-out, 25 training IDs, 20-token suffix')
    print('Saved Task 2 attack successes (reference only):')
    print(summary.to_string())

if __name__=='__main__': main()
