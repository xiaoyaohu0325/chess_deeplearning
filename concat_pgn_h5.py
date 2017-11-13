import h5py
import os
import argparse
import sys


def hdf5_cancat(hdf5_files, output):
    # initialise the output
    tmp_file = os.path.join(os.path.dirname(output), ".tmp." + os.path.basename(output))
    combined = h5py.File(tmp_file, 'w')
    try:
        dt = h5py.special_dtype(vlen=bytes)
        combined.require_dataset(
            name='draw',
            dtype=dt,
            shape=(0, ),
            maxshape=(None, ),
            chunks=True,
            compression="lzf")
        combined.require_dataset(
            name='white',
            dtype=dt,
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            compression="lzf")
        combined.require_dataset(
            name='black',
            dtype=dt,
            shape=(0,),
            maxshape=(None,),
            chunks=True,
            compression="lzf")

        white = combined["white"]
        black = combined["black"]
        draw = combined["draw"]

        for filename in hdf5_files:
            fileread = h5py.File(filename, 'r')
            white_data = fileread['white']
            black_data = fileread['black']
            draw_data = fileread['draw']

            w_start = len(white)
            w_end = w_start + len(white_data)
            white.resize((w_end,))
            white[w_start:w_end] = white_data

            b_start = len(black)
            b_end = b_start + len(black_data)
            black.resize((b_end,))
            black[b_start:b_end] = black_data

            d_start = len(draw)
            d_end = d_start + len(draw_data)
            draw.resize((d_end,))
            draw[d_start:d_end] = draw_data

            fileread.close()

        combined.close()
        os.rename(tmp_file, output)
    except Exception as e:
        os.remove(tmp_file)
        raise e


def run_cancat(cmd_line_args=None):
    """Run cancatenations. command-line args may be passed in as a list
    """
    parser = argparse.ArgumentParser(
        description='Cancatenate the generated hdf5 files',
        epilog="A directory containing the hdf5 files is needed")
    parser.add_argument("--outfile", "-o", help="Destination to write data (hdf5 file)", required=True)
    parser.add_argument("--recurse", "-R", help="Set to recurse through directories searching for HDF5 files",
                        default=False, action="store_true")
    parser.add_argument("--directory", "-d", help="Directory containing HDF5 files to process")

    if cmd_line_args is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(cmd_line_args)

    if args.directory:
        if args.recurse:
            files = list(_walk_all_hdf5s(args.directory))
        else:
            files = list(_list_hdf5s(args.directory))
    else:
        files = list((f.strip() for f in sys.stdin if _is_hdf5(f)))

    hdf5_cancat(files, args.outfile)


def _is_hdf5(fname):
    return fname.strip()[-5:] == ".hdf5" or fname.strip()[-3:] == ".h5"


def _walk_all_hdf5s(root):
    """a helper function/generator to get all hdf5 files in subdirectories of root
    """
    for (dirpath, dirname, files) in os.walk(root):
        for filename in files:
            if _is_hdf5(filename):
                # yield the full (relative) path to the file
                yield os.path.join(dirpath, filename)


def _list_hdf5s(path):
    """helper function to get all hdf5 files in a directory (does not recurse)
    """
    files = os.listdir(path)
    return (os.path.join(path, f) for f in files if _is_hdf5(f))


if __name__ == '__main__':
    run_cancat()
