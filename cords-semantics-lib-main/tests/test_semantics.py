import unittest
from rdflib import Graph, Namespace, RDF
from rdflib.compare import graph_diff, isomorphic
from cords_semantics.semantics import MlSemanticManager 
import json

# Assuming MlSemanticManager is defined in the same file or properly imported
class TestMlSemanticManager(unittest.TestCase):
    def setUp(self):
        self.manager = MlSemanticManager("data/cordsml.rdf")

    def test_initial_namespace_uris(self):
        """Test initial URIs are correctly set."""
        self.assertEqual(self.manager.cords_namespace_uri, "http://www.w3.org/ns/cordsns#")
        self.assertEqual(self.manager.scenario_uri, "http://example.org#")

    def test_set_cords_namespace_uri(self):
        """Test setting the CORDS namespace URI."""
        new_uri = "http://www.w3.org/ns/cordsns#"
        self.manager.set_cords_namespace_uri(new_uri)
        self.assertEqual(self.manager.cords_namespace_uri, new_uri)

    def test_set_scenario_namespace_uri(self):
        """Test setting the scenario namespace URI."""
        new_uri = "http://example.org#"
        self.manager.set_scenario_namespace_uri(new_uri)
        self.assertEqual(self.manager.scenario_uri, new_uri)

    def test_create_model_semantics(self):
        """Test creating the RDF graph from metadata."""
        meta_data = {'ModelEvaluation': [{'type': 'modelEvaluation_test_MAE', 'specifiedBy': 'test_MAE', 'hasValue': '1.5492494294763557'},
                                          {'type': 'modelEvaluation_test_R2_Score', 'specifiedBy': 'test_R2_Score', 'hasValue': '0.858882955307253'}, 
                                          {'type': 'modelEvaluation_test_RMSE', 'specifiedBy': 'test_RMSE', 'hasValue': '2.7181378412972808'}], 
                                          'Implementation': [{'type': 'python'}], 'Run': [{'type': '5fc5acb21557479eb3c8146bede17732', 
                                                                                           'executes': 'K-NeighborsRegressor'}], 
                                                                                           'Software': [{'type': 'sklearn'}]}
        
        expected_graph = Graph()

        expected_graph.parse("tests/output.rdf", format='application/rdf+xml')

        created_graph = self.manager.create_model_semantics(meta_data)


        # # Check if the created graph is isomorphic to the expected graph
        self.assertTrue(isomorphic(created_graph, expected_graph))

    def test_convert_to_json_ld(self):
        """Test creating the json-ld object."""
        meta_data = {'ModelEvaluation': [{'type': 'modelEvaluation_test_MAE', 'specifiedBy': 'test_MAE', 'hasValue': '1.5492494294763557'},
                                          {'type': 'modelEvaluation_test_R2_Score', 'specifiedBy': 'test_R2_Score', 'hasValue': '0.858882955307253'}, 
                                          {'type': 'modelEvaluation_test_RMSE', 'specifiedBy': 'test_RMSE', 'hasValue': '2.7181378412972808'}], 
                                          'Implementation': [{'type': 'python'}], 'Run': [{'type': '5fc5acb21557479eb3c8146bede17732', 
                                                                                           'executes': 'K-NeighborsRegressor'}], 
                                                                                           'Software': [{'type': 'sklearn'}]}
        
        created_graph = self.manager.create_model_semantics(meta_data)

        json_ld = self.manager.convert_to_json_ld()

        expected_json_ld = json.loads("tests/json_ld_output.json")

        self.assertEqual(json_ld, expected_json_ld)


if __name__ == '__main__':
    unittest.main()