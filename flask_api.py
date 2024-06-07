from flask import Flask, request, send_file, jsonify
import mlflow
import shutil, os
from mlflow.tracking import MlflowClient

app = Flask(__name__)

@app.route('/get_model', methods=['GET'])
def get_model():
    run_id = request.args.get('run_id')
    model_name = request.args.get('model_name')
    
    # Construct the local path to the model artifact
    model_artifact_path = os.path.join("mlartifacts/0", run_id, "artifacts", model_name)
    client = MlflowClient()
    
    try:
        if not os.path.exists(model_artifact_path):
            raise FileNotFoundError(f"Model artifact not found at {model_artifact_path}")

        # Fetch the tags associated with the run
        tags = client.get_run(run_id).data.tags

        # Check if the path is a directory or file and handle accordingly
        if os.path.isdir(model_artifact_path):
            # If it's a directory, zip it first (you may need to import shutil)
            import shutil
            shutil.make_archive("model_archive", 'zip', model_artifact_path)
            model_path = "model_archive.zip"
        else:
            model_path = model_artifact_path
         # Send the model file and tags as part of the response
        return send_file(model_path, as_attachment=True, download_name=os.path.basename(model_path))
    except Exception as e:
        return jsonify({"message": str(e)}), 500

if __name__ == "__main__":
    app.run(port=5001, debug=True)
