import argparse

def get_params():
    # parse parameters
    parser = argparse.ArgumentParser(description="Cross-lingual Task-Oriented Dialog")
    parser.add_argument("--exp_name", type=str, default="default", help="Experiment name")
    parser.add_argument("--logger_filename", type=str, default="multilingual_dst.log")
    parser.add_argument("--dump_path", type=str, default="experiments", help="Experiment saved root path")
    parser.add_argument("--exp_id", type=str, default="1", help="Experiment id")
    parser.add_argument("--trans_lang", type=str, default="", help="Choose a language to transfer")
    
    # binarize data
    parser.add_argument("--vocab_path_en", type=str, default="data/dst/dst_vocab/vocab.en", help="Path of vocabulary")
    parser.add_argument("--vocab_path_trans", type=str, default="", help="Path of vocabulary")

    parser.add_argument("--ontology_class_path", type=str, default="data/dst/dst_data/ontology_classes.json", help="Path of ontology classes")
    parser.add_argument("--ontology_mapping_path", type=str, default="data/dst/dst_data/ontology-mapping.json", help="Path of ontology mapping")

    # model parameters
    parser.add_argument("--bidirection", default=False, action="store_true", help="Bidirectional lstm")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate for lstm")
    parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=200, help="Hidden layer dimension")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0, help="Weight decay")
    parser.add_argument("--epoch", type=int, default=300, help="Number of epoch")
    parser.add_argument("--emb_file_en", type=str, default="data/dst/dst_emb/embedding_en.npy", help="Path of pretrained embeddings in English")
    parser.add_argument("--emb_file_trans", type=str, default="data/dst/dst_emb/embedding_de.npy", help="Path of pretrained embeddings in transfer language")
    parser.add_argument("--ontology_emb_file", type=str, default="data/dst/dst_data/ontology_embeddings_en.npy", help="Path of ontology embedding file")
    parser.add_argument("--gate_size", type=int, default=300, help="Gate size (should be same as embedding dimension)")
    parser.add_argument("--early_stop", type=int, default=10, help="No improvement after several epoch, we stop training")
    # number of classes for slots and request
    parser.add_argument("--food_class", type=int, default=76, help="the number of classes for food slot (include none)")
    parser.add_argument("--price_range_class", type=int, default=5, help="the number of classes for price range slot (include none)")
    parser.add_argument("--area_class", type=int, default=7, help="the number of classes for area slot (include none)")
    parser.add_argument("--request_class", type=int, default=7, help="the number of classes for request")

    # mix languages training
    parser.add_argument("--mix_train", default=False, action="store_true", help="Mix language training")
    parser.add_argument("--mapping_for_mix", type=str, default="data/dst/dst_vocab/en2de_onto_for_mix.dict", help="mapping for mix language training")

    # run nlu dataset
    parser.add_argument("--run_nlu", default=False, action="store_true", help="Run NLU dataset")
    parser.add_argument("--n_layer", type=int, default=2, help="Number of lstm layer")
    parser.add_argument("--num_intent", type=int, default=12, help="Number of intent in the dataset")
    parser.add_argument("--num_slot", type=int, default=24, help="Number of slot in the dataset")
    parser.add_argument("--clean_txt", default=False, action="store_true", help="Clean text if store true")
    parser.add_argument("--filtered", default=False, action="store_true", help="filter attention selected words data samples")
    parser.add_argument("--filter_scale", type=str, default="20", help="filter based on how many attention selected words")

    params = parser.parse_args()

    return params
