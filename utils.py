from langchain_community.chat_models import ChatOpenAI
import numpy as np
import traceback
from rule_ordering import find_path
import os, json, re
import pandas as pd
import warnings 


warnings.filterwarnings("ignore")

def mean(s):
    return sum(s) / len(s)

prompts_mode = {'Detect': {
            'template': "Give me a function in python that will take each row as input and return a list of the values that confirm this error. Each column should have its own entry in the output. Example of output: [{{'column': <col_name>, 'index': <index_no>}}]",
            'system': "You are an assistant that will help in error detection in tabular data."
            }, 
           'Correct': {
            'template': "Give me a function in python that will take each row as input and return a list of imputed values. Perform this action only on missing values. Each column should have its own entry in the output. Example of output: [{{'column': <col_name>, 'index': <index_no>, 'value': <imputed_value>}}]",
            'system': "You are an assistant that will help in data imputation on missing values in tabular data."
               },
           'Decouple': {
               'template': "The following function affects more than one columns. You must create equivalent separate functions, so each function affects only one column. \nORIGINAL FUNCTION:\n{}\n",
               'system': 'You are an assistant that will help decouple functions that affect more than one columns.'
               },
           'Relation': {
               'template': "Given the following function, you should give me a list of affected columns and their dependent columns. \nEXAMPLE:\n#Function\ndef foo(df):\ndf['Column C'] = df['Column A'] + df['Column B'] \n\n#Output\n{{'Column C': ['Column A', 'Column B']}}\n\nA column can also be dependent on itself, e.g. if using the average of the column C to fill missing values of Column C, then {{'Column C': ['Column C']}}\n.Return only this list, no text or comments.\n\nFunction: def foo(df):\n{}\n\n#Output:",
               'system': 'You are a helpful assistant in detecting affected columns from Python functions.'
               },
           'Grouping': {
               'template': "We have a dataframe with the following columns:\n\n{}\n\nFind me groups of columns that might be related. Example of output: [('column1', 'column2', 'column3'), ('column4', 'column5'), ...]",
               'system': 'You are a helpful assistant in detecting groups of columns that might be related, based on names or domains.'
               },
           'Rules': {
               'template': "We have a dataframe with the following columns:\n\n{}\n\nGenerate a list of rule(s) that involve these columns, regarding {}, i.e. the task of {}. The rule(s) should be in natural language. Example of output: [rule1, rule2, ...]",
               'system': 'You are a helpful assistant in generating rules for columns that might be related, based on their names or domains.'
               }             
           
           }

class CodeGenerator():
    def __init__(self):
        self.query_cache = {}
        if os.path.exists('query_cache.jsonl'):
            with open('query_cache.jsonl') as f:
                for line in f:
                    j = json.loads(line)
                    self.query_cache[(j['system'], j['human'])] = j['response']
    
    def append_response(self, system, human, response):
        self.query_cache[(system, human)] = response
        
        with open('query_cache.jsonl', 'a') as f:
            f.write(json.dumps({'system': system, 
                               'human': human,
                               'response': response}) + '\n')
            
    def enable_function(self, function):
        foo_name = self.extract_function_name(function)         
    
        namespace = {}
        exec(function, namespace)

        foo = namespace[foo_name]

        return foo
            
    def split_functions(self, code_str):
        parts = re.split(r'(?=^def\s+)', code_str, flags=re.MULTILINE)
        
        # Extract top imports
        imports = []
        for line in code_str.splitlines():
            if line.startswith(("import", "from")):
                imports.append(line)
            else:
                break
        import_block = "\n".join(imports)
    
        functions = []
        for part in parts:
            part = part.strip()
            
            if not part.startswith("def"):
                continue
            
            # Prepend imports once
            if import_block:
                part = import_block + "\n\n" + part
            
            functions.append(part)
    
        return functions
    
    def extract_function_code(self, response):
        foo_pattern = r"```python(.*?)```"
        match = re.search(foo_pattern, response, re.DOTALL)
        
        code_str = match.group(1).strip()
        
        #adding missing imports
        imports = re.findall(r'^\s*import\b.*$', response, flags=re.MULTILINE)
        for imp in imports:
            code_str = imp + '\n' + code_str
        
        #adding manual imports (pandas)
        if "pd." in code_str and "import pandas" not in code_str:
            code_str = "import pandas as pd\n"+code_str    
            
        return code_str
    
    def extract_function_name(self, response):
        def_pattern = r"def (.*?)\("
        match = re.search(def_pattern, response, re.DOTALL)
        foo_name = match.group(1).strip()   
        
        return foo_name
    
    def extract_affected_columns(self, response):
        col_pattern = "({.*?})"
        match = re.search(col_pattern, response, re.DOTALL)
        
        affected_columns = match.group(1).strip()
        affected_columns = eval(affected_columns)    
        return affected_columns
    
    
    def query_llm(self, human_prompt, system_prompt, model, endpoint, token):
        if (system_prompt, human_prompt) in self.query_cache:
            return self.query_cache[(system_prompt, human_prompt)]
        messages = [("system", system_prompt),
                    ("human", human_prompt)]
        
        llm = ChatOpenAI(model_name=model, 
                         openai_api_base=endpoint,
                         openai_api_key=token, 
                         temperature=0.0, 
                         max_tokens=2048,
                         timeout=120)
        response = llm.invoke(messages)
        response = response.content
        
        self.append_response(system_prompt, human_prompt, response)
        return response
    
    def generate_code_dumb(self, rules, mode):
        """
        Generates a list of function stubs based on the rules and mode.
        """
        functions = []
    
        for i, rule in enumerate(rules):
            func_name = f"{mode.lower()}_rule_{i+1}"
            func_code = f"""
    def {func_name}(df):
        # Rule: {rule}
        pass
    """
            functions.append({"name": func_name, "code": func_code})
    
        return functions
    
    
    def generate_rules(self, df, mode, model, endpoint, token):
        """
        Generates a list of function stubs based on the rules and mode.
        """
        log = {'functions': [], 'values': [], 'prompts': []}
        
        dtypes = {}
        for col in df.columns:
            dtypes[col] = df[col].dtype
    
        profile_str = ""
        for col in dtypes: #TODO: change with profiles
            profile_str += f"\t{col}: type -> {dtypes[col]}\n"
    
        prompt_template = prompts_mode['Grouping']['template']
        system_template = prompts_mode['Grouping']['system']
    
        print('STARTING COLUMN GROUPING:')
        prompt = prompt_template.format(profile_str)
        system = system_template
        response = self.query_llm(prompt, system, model, endpoint, token)
        print(response)
        
        task = 'Detecting errors' if mode=='Detect' else 'Imputating values on missing data'
        prompt_template = prompts_mode['Rules']['template']
        system_template = prompts_mode['Rules']['system']
        print('STARTING RULE GENERATION:')
        groups = re.findall("\(.*?\)", response, re.DOTALL)
        if len(groups) == 0:
            groups = re.findall("\[.*?\]", response, re.DOTALL)
        
        total_rules = []
        for nog, group in enumerate(groups):
            split_group = eval(group)
            print('Parsing group {} of {}'.format(nog, len(groups)))
            # print(group, split_group)
            if type(split_group) == str: # 1-set group
                split_group = [split_group]
            
            profile_str = ""
            for col in split_group: #TODO: change with profiles
                profile_str += f"\t{col}: type -> {dtypes[col]}\n"
                
            prompt = prompt_template.format(profile_str, mode, task)
            system = system_template
            response = self.query_llm(prompt, system, model, endpoint, token)
            
            # rules = re.findall(r"Rule\s+\d+:\s*(.*?)(?=\n\s*Rule\s+\d+:|\Z)",
            #                    response, flags=re.DOTALL)
            rules = re.findall(
                r"(?:\d+\.\s*)?Rule\s+\d+:\s*(.*?)(?=(?:\d+\.\s*)?Rule\s+\d+:|\Z)",
                response,
                flags=re.DOTALL
            )

            rules = [r.strip().replace("\n", " ") for r in rules]

            total_rules += rules
            
        return total_rules
        
        
    
        # try:
        #     code_str = self.extract_function_code(response)
        #     foo = self.enable_function(code_str)
        #     values = foo(temp_df)
        #     log['prompts'].append((system, prompt))
        #     log['functions'].append(code_str)
        #     log['values'] += [values]
        #     break # successful try
        # except Exception as e:
        #     exception = f"{type(e).__name__}: {e}"
        #     print('\tError in this function.', exception)    
    
    
    def generate_code(self, df, rules, mode, model, endpoint, token, 
                      rules_order="Linear", notries=10):
        """
        Generates a list of function stubs based on the rules and mode.
        """
        log = {'functions': [], 'values': [], 'prompts': []}
        
        dtypes = {}
        for col in df.columns:
            dtypes[col] = df[col].dtype
    
        profile_str = ""
        for col in dtypes: #TODO: change with profiles
            profile_str += f"\t{col}: type -> {dtypes[col]}\n"
    
        prompt_template = "We have a dataframe with the following columns:\n\n{}\n\nThe following rule:{}\n"
        prompt_template += prompts_mode[mode]['template']
        system_template = prompts_mode[mode]['system']
    
        print('STARTING RULE Parsing:')
        for nor, rule in enumerate(rules):
            print('\tParsing Rule: {} ({}/{})'.format(rule, nor+1, len(rules)))
            exception = ""
            prompt = prompt_template.format(profile_str, rule)
            system = system_template
            for tries in range(notries):
                temp_df = df.copy()
                if tries > 0:
                    if tries == 1: #first error:
                        system += "\nKNOWN FAILURE MODE:\n"
                    system += "- {}\n".format(exception)
        
                response = self.query_llm(prompt, system, model, endpoint, token)
        
                try:
                    code_str = self.extract_function_code(response)
                    foo = self.enable_function(code_str)
                    values = foo(temp_df)
                    log['prompts'].append((system, prompt))
                    log['functions'].append(code_str)
                    log['values'] += [values]
                    break # successful try
                except Exception as e:
                    exception = f"{type(e).__name__}: {e}"
                    print('\tError in this function.', exception)
                        
    
        optimal_path = None
        if rules_order == 'Advanced' and mode == 'Correct': 
            print('STARTING RELATION ANALYSIS IN FUNCTIONS:')
            new_log = {'values': [], 'functions': [], 'prompts': []}
            edges = []
            
            for nof, function in enumerate(log['functions']):
                print(f"\tParsing Function {nof}")
                print(function)
                response = self.query_llm(prompts_mode['Relation']['template'].format(function), 
                                          prompts_mode['Relation']['system'], 
                                          model, endpoint, token)
                
                try:
                    affected_columns = self.extract_affected_columns(response)
                    print('\tAffected columns: ', affected_columns)
                    
                    if len(affected_columns) > 1: #decouple functions
                        print('\tDecoupling Function')
                        response = self.query_llm(prompts_mode['Decouple']['template'].format(function), 
                                                  prompts_mode['Decouple']['system'], 
                                                  model, endpoint, token)
                        
                        try:
                            new_functions = self.extract_function_code(response)
                            new_functions = self.split_functions(new_functions)
                            
                            for noff, new_function in enumerate(new_functions):
                                print(f"\t\tParsing inner Function {noff}")
                                temp_df = df.copy()
                                foo = self.enable_function(new_function)
                                values = foo(temp_df)
                                new_log['prompts'].append(log['prompts'][nof])
                                new_log['functions'].append(new_function)
                                new_log['values'] += [values]
                                
                                response = self.query_llm(prompts_mode['Relation']['template'].format(new_function), 
                                                          prompts_mode['Relation']['system'], 
                                                          model, endpoint, token)
                                
                                affected_columns = self.extract_affected_columns(response)
                                print('\t\tAffected columns: ', affected_columns)

                                for target, sources in affected_columns.items():
                                    for source in sources:
                                        edges.append({'target': target, 'source': source,
                                                      'weight': len(values), 'function': len(new_log['functions'])-1})

                                
                        except Exception as e:
                            exception = f"{type(e).__name__}: {e}"
                            print('\tError in this function.', exception)                        
    
                    # print(new_log)
                    else:
                        new_log['prompts'].append(log['prompts'][nof])
                        new_log['functions'].append(log['functions'][nof])
                        new_log['values'] += [log['values'][nof]]
                        for target, sources in affected_columns.items():
                            for source in sources:
                                edges.append({'target': target, 'source': source,
                                              'weight': len(log['values'][nof]), 'function': len(new_log['functions'])-1})                        
    
                except Exception as e:
                    traceback.print_exc()
                    continue
            
                print("\tIteration {}: {}".format(nof, edges))
    
            log = new_log
            edges = [e for e in edges if e['weight']>0] #clean from redundant edges
            optimal_path = find_path(edges)
        
        # Deduplicate errors and keep score for each cell
        if mode == 'Detect':
            unique_errors = {}
            for temp_list in log['values']:
                for e in temp_list: 
                    if (e['index'], e['column']) not in unique_errors:
                        unique_errors[(e['index'], e['column'])] = 0
                    unique_errors[(e['index'], e['column'])] += 1
        
            result_df = df.copy()
            # anomalies = []
            for e, val in unique_errors.items():
                result_df.loc[e[0], e[1]] = np.nan
                # anomalies.append({'row': e[0], 'col': e[1], 'score': val})
            keys = unique_errors.keys()
        else: 
            
            keys = []
            if optimal_path: #advanced rule ordering
                result_df = df.copy()
                for path in optimal_path: # path -> (source, target, info)
                    index = path[2]['function']
                    foo = self.enable_function(log['functions'][index])
                    results = foo(result_df)
                    column = path[1]
                    
                    for res in results:
                        if pd.isna(res['value']):
                            continue
                        # if res['value'] == result_df.loc[res['index'], column]:
                        #     continue    
                        keys.append((res['index'], column))
                        result_df.loc[res['index'], column] = res['value']
                        
                            
            else:
                result_df = df.copy()
                for function in log['functions']:
                    foo = self.enable_function(function)
                    results = foo(result_df)

                    for res in results:
                        if pd.isna(res['value']):
                            continue
                        # if res['value'] == result_df.loc[res['index'], res['column']]:
                        #     continue
                        keys.append((res['index'], res['column']))
                        result_df.loc[res['index'], res['column']] = res['value']
            
        output = {'df': result_df, 'functions': log['functions'], 
                  'indices': keys, 'values': log['values']}
        
        if optimal_path:
            output['optimal_path'] = optimal_path
            output['edges'] = edges
            
        return output
