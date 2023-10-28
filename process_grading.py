import json
import traceback

import pandas as pd
from tabulate import tabulate
import io
import os.path as osp

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 60)

# Sample Markdown table

if __name__ == "__main__":

    ROOT = "/Users/ahren/Workspace/Packages/ChatPaper"
    ROOT = "/Users/ahren/Library/CloudStorage/OneDrive-GeorgiaInstituteofTechnology/Courses/CSE8803DSN/Grading"

    ASSIGNMENT_NUM = 3
    results_d = json.load(open(osp.join(ROOT, f'grades_HW{ASSIGNMENT_NUM}.json'), 'r', encoding='utf-8'))

    student_name2df = {}

    for filename, results in results_d.items():

        student_name = filename.split('_')[0]
        print(f"Student name: {student_name}")

        # Use StringIO to simulate a file-like object
        tsv_io = io.StringIO(results)

        # Load the TSV data into a pandas DataFrame
        df = pd.read_csv(tsv_io, delimiter='\t')

        print(df)

        student_name2df[student_name] = df




    with pd.ExcelWriter(f"{ROOT}/grades_hw{ASSIGNMENT_NUM}.xlsx", engine='openpyxl', mode='w') as writer:

        for student_name, df_to_save in student_name2df.items():
            sheet_name = f"Student {student_name}"
            print(f"Saving {sheet_name}")
            df_to_save.to_excel(writer, sheet_name=f'{student_name}',
                        index=False)

    with pd.ExcelWriter(f"{ROOT}/grades_all.xlsx", engine='openpyxl', mode='a',
                        if_sheet_exists='replace') as writer:
        df = pd.DataFrame(columns=['name', 'Points'])

        for student_name, student_df in student_name2df.items():

            try:

                total_points = student_df['Point Rewarded'].values[-1].sum()

            except:
                traceback.print_exc()

            df = pd.concat([df, pd.Series({'name': student_name, 'Points': total_points}).to_frame().T])


        df.set_index("name", inplace=True)

        df.to_excel(writer, sheet_name=f'Assignment{ASSIGNMENT_NUM}', index=False)



    df_summary = pd.read_excel(f"{ROOT}/grades_summary.xlsx")

    df_summary.set_index("name", inplace=True)

    df_summary = pd.concat([df_summary, df.rename({"Points": f"HW{ASSIGNMENT_NUM}"}, axis=1)], axis=1)

    df_summary.to_excel(f"{ROOT}/grades_summary.xlsx", index=True)












