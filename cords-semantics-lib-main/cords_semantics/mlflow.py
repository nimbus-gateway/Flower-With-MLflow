from mlflow.tracking import MlflowClient


def convert_tags_to_dictionary(input_tags_list):
    """
    Transforms a list of dictionaries of tags in a structured dictionary
    based on the hierarchy implied by the keys.

    Each key in the input dictionary should be formatted as 'prefix.Property.Subproperty',
    which will be transformed into a dictionary where each 'Property' is a key that contains
    a list of dictionaries for its sub-properties.

    Args:
    input_tags_list (list): A list of dictionaries. A dictionary consist flat keys that represent hierarchical
                            properties using dot notation.

    Returns:
    dict: A nested dictionary reflecting the hierarchical structure of input keys.
    """
    output_tags_dict = {}
    for input_tags_dict in input_tags_list:
        for key, value in input_tags_dict.items():
            key_parts = key.split(".")
            
            # Handle the base level (assuming always at least two parts, otherwise skip the entry)
            if len(key_parts) < 2:
                continue
            
            main_key = key_parts[1]
            
            if len(key_parts) == 2:
                # Initialize the list for main_key if it doesn't exist
                if main_key not in output_tags_dict:
                    output_tags_dict[main_key] = []
                # Append a new dictionary with 'type' property
                output_tags_dict[main_key].append({"type": value})
            
            elif len(key_parts) == 3:
                # Add additional properties to the last dictionary in the list if possible
                if main_key in output_tags_dict and output_tags_dict[main_key]:
                    last_item = output_tags_dict[main_key][-1]
                    last_item[key_parts[2]] = value
                else:
                    # Handle the case where there's no entry to add properties to
                    if main_key not in output_tags_dict:
                        output_tags_dict[main_key] = []
                    output_tags_dict[main_key].append({key_parts[2]: value})
    
    return output_tags_dict



def extract_mlflow_semantics(run_id):
    """
    Extract semantic data from an MLflow run based on specific tag and metric patterns.

    This function retrieves semantic data by identifying specific tags and metrics from a given MLflow run.
    It processes these elements to construct a dictionary of tag values containing 'cords' and a list of metric dictionaries
    that encapsulate model evaluation details, formatted according to the 'cords' namespace.

    Args:
        run_id (str): The unique identifier for the MLflow run.

    Returns:
        list: A list containing dictionaries of metrics formatted with semantic data and a dictionary of selected tags.

    Raises:
        MlflowException: If there is an error in retrieving the run from MLflow.
    """
    from mlflow.tracking import MlflowClient

    client = MlflowClient()
    run = client.get_run(run_id)
    tags_semantic_dict = {}
    semantic_list = []
    
    tags = run.data.tags
    for tag in tags:
        if 'cords' in tag:
            tags_semantic_dict[tag] = tags[tag]

    metrics = run.data.metrics
    for metric in metrics:
        metric_dict = {}
        metric_dict["cords.ModelEvaluation"] = "modelEvaluation_" + metric
        metric_dict["cords.ModelEvaluation.specifiedBy"] = metric
        metric_dict["cords.ModelEvaluation.hasValue"] = str(metrics[metric])
        semantic_list.append(metric_dict)
        
    semantic_list.append(tags_semantic_dict)

    return semantic_list