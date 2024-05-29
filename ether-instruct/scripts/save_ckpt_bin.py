


if not model_path.exists() or force_conversion:
    copy_config_files(source_dir=checkpoint_dir, out_dir=out_dir)
    convert_lit_checkpoint(checkpoint_dir=checkpoint_dir, output_dir=out_dir)

    # Hack: LitGPT's conversion doesn't save a pickle file that is compatible to be loaded with
    # `torch.load(..., weights_only=True)`, which is a requirement in HFLM.
    # So we're `torch.load`-ing and `torch.sav`-ing it again to work around this.
    state_dict = torch.load(out_dir / "model.pth")
    torch.save(state_dict, model_path)
    os.remove(out_dir / "model.pth")