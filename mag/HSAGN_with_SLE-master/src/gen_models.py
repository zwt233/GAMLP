from models import SAGN, HSAGN, NARS
from new_model import NARS_gmlp
def get_model(in_feats, n_classes, stage, relations_set, args):
    num_hops = args.K + 1
    label_in_feats = n_classes
    use_labels = args.use_labels and ((not args.inductive) or stage > 0)
    use_features = not args.avoid_features
    batch_norm = not args.no_batch_norm
    if args.model == "hsagn":
        model = HSAGN(in_feats, args.num_hidden, n_classes, label_in_feats, args.K,
                        args.mlp_layer, args.num_heads, relations_set,
                        dropout=args.dropout,
                        input_drop=args.input_drop,
                        attn_drop=args.attn_drop,
                        use_labels=use_labels,
                        use_features=use_features)

    if args.model == "nars_sagn":
        model = NARS(in_feats, args.num_hidden, n_classes, label_in_feats, num_hops,
                args.multihop_layers, args.mlp_layer, args.num_heads, relations_set,
                clf="sagn",
                relu=args.relu,
                batch_norm=batch_norm,
                dropout=args.dropout,
                input_drop=args.input_drop,
                attn_drop=args.attn_drop,
                use_labels=use_labels,
                use_features=use_features)
    if args.model == "nars_sign":
        model = NARS(in_feats, args.num_hidden, n_classes, label_in_feats, num_hops,
                args.multihop_layers, args.mlp_layer, args.num_heads, relations_set,
                clf="sign",
                dropout=args.dropout,
                input_drop=args.input_drop,
                attn_drop=args.attn_drop,
                use_labels=use_labels,
                use_features=use_features)
    if args.model=='nars_gmlp':
       model = NARS_gmlp(in_feats, args.num_hidden, n_classes, label_in_feats, num_hops,args.multihop_layers, args.mlp_layer, args.num_heads, relations_set, clf="gmlp", dropout=args.dropout,input_drop=args.input_drop,attn_drop=args.attn_drop,use_labels=use_labels,use_features=use_features)
    return model
