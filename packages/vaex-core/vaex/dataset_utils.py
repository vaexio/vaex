# -*- coding: utf-8 -*-
# from vaex import dataset
import vaex.dataset
from optparse import OptionParser
from mab.utils.progressbar import ProgressBar
import sys
import h5py
import numpy as np


def merge(output_filename, datasets_list, datasets_centering=None, sort_property=None, order_column_name=None, ascending=True):
    # datasets = list(datasets)
    if sort_property:
        datasets_list.sort(key=lambda datasets: datasets[0].variables[sort_property], reverse=not ascending)
        datasets_centering.sort(key=lambda dataset: dataset.variables[sort_property], reverse=not ascending)

    h5output = h5py.File(output_filename, "w")
    example_dataset = datasets_list[0][0]
    max_length = max([sum(len(dataset) for dataset in datasets) for datasets in datasets_list])
    # counts =

    # for datasets in datasets:
    # max_length = sum( counts )
    shape = (len(datasets_list), max_length)
    print(("shape of new arrays will be", shape, max_length))
    if 0:
        for dataset1 in datasets:
            for dataset2 in datasets:
                if dataset1 != dataset2:
                    if len(dataset1) != len(dataset2):
                        print((dataset1.name, "is of length", len(dataset1), "but", dataset2.name, "is of length", len(dataset2)))
                        sys.exit(1)

    for column_name in example_dataset.column_names:
        d = h5output.require_dataset("/columns/" + column_name, shape=shape, dtype=example_dataset.columns[column_name].dtype.type, exact=True)
        d[0, 0] = example_dataset.columns[column_name][0]  # ensure the array exists
    # each float propery will be a new axis in the merged file (TODO: int and other types?)
    for property_name in list(example_dataset.variables.keys()):
        property = example_dataset.variables[property_name]
        if isinstance(property, float):
            d = h5output.require_dataset("/axes/" + property_name, shape=(len(datasets_list),), dtype=np.float64, exact=True)
            d[0] = 0.  # make sure it exists
    # close file and open it again with our interface
    h5output.close()
    dataset_output = vaex.dataset.Hdf5MemoryMapped(output_filename, write=True)

    progressBar = ProgressBar(0, len(datasets_list) - 1)

    if 0:
        idmap = {}
        for index, dataset in enumerate(datasets_list):
            ids = dataset.columns["ParticleIDs"]
            for id in ids:
                idmap[id] = None
        used_ids = list(idmap.keys())
        print((sorted(used_ids)))

    particle_type_count = len(datasets_list[0])
    for index, datasets in enumerate(datasets_list):

        centers = {}
        if datasets_centering is not None:
            cols = datasets_centering[index].columns
            # first rought estimate
            for name in "x y z vx vy vz".split():
                indices = np.argsort(cols["Potential"])[::1]
                indices = indices[:10]
                # centers[name] = cols[name][np.argmin(cols["Potential"])] #.mean()
                centers[name] = cols[name].mean()
                print(("center", centers[name]))

            if 0:
                # if column_name in "x y z".split():

                # now sort by r
                r = np.sqrt(np.sum([(cols[name] - centers[name])**2 for name in "x y z".split()], axis=0))
                indices = np.argsort(r)
                indices = indices[:len(indices) / 2]  # take 50%

                for name in "x y z".split():
                    centers[name] = cols[name][indices].mean()

                # sort by v
                v = np.sqrt(np.sum([(cols[name] - centers[name])**2 for name in "vx vy vz".split()]))
                indices = np.argsort(r)
                indices = indices[:len(indices) / 2]  # take 50%

                for name in "vx vy vz".split():
                    centers[name] = cols[name][indices].mean()

        for column_name in datasets[0].column_names:
            column_output = dataset_output.rank1s[column_name]
            column_output[index, :] = np.nan  # first fill with nan's since the length of the output column may be larger than that of individual input datasets

            for property_name in list(datasets[0].variables.keys()):
                property = datasets[0].variables[property_name]
                if isinstance(property, float):
                    # print "propery ignored: %r" % property
                    # print "propery set: %s %r" % (property_name, property)
                    dataset_output.axes[property_name][index] = property
                else:
                    # print "propery ignored: %s %r" % (property_name, property)
                    pass

            center = 0

            output_offset = 0
            # merge the multiple datasets into the one column
            for particle_type_index in range(particle_type_count):
                dataset = datasets[particle_type_index]
                # print len(dataset), output_offset

                column_input = dataset.columns[column_name]
                if order_column_name:
                    order_column = dataset.columns[order_column_name]
                else:
                    order_column = None
                # print dataset.name, order_column, order_column-order_column.min()
                i1, i2 = output_offset, output_offset + len(dataset)
                if order_column is not None:
                    column_output[index, order_column - order_column.min()] = column_input[:] - centers.get(column_name, 0)
                else:
                    column_output[index, i1:i1 + len(dataset)] = column_input[:] - centers.get(column_name, 0)
                output_offset += len(dataset)
            # print "one file"
            progressBar.update(index)


if __name__ == "__main__":
    usage = "use the source luke!"
    parser = OptionParser(usage=usage)

    # parser.add_option("-n", "--name",
    # help="dataset name [default=%default]", default="data", type=str)
    parser.add_option("-o", "--order", default=None, help="rows in the input file are ordered by this column (For gadget: ParticleID)")
    parser.add_option("-t", "--type", default=None, help="file type")
    parser.add_option("-p", "--particle-types", default=None, help="gadget particle type")
    parser.add_option("-c", "--center_type", default=None, help="gadget centering type")
    # parser.add_option("-i", "--ignore", default=None, help="ignore errors while loading files")
    parser.add_option("-r", "--reverse", action="store_true", default=False, help="reverse sorting")
    parser.add_option("-s", "--sort",
                      help="sort datasets by propery [by default it will be the file order]", default=None, type=str)
    (options, args) = parser.parse_args()
    inputs = args[:-1]
    output = args[-1]
    print(("merging:", "\n\t".join(inputs)))
    print(("to:", output))
    if options.type is None:
        print("specify file type --type")
        parser.print_help()
        sys.exit(1)
    # dataset_type_and_options = options.format.split(":")
    # dataset_type, dataset_options = dataset_type_and_options[0], dataset_type_and_options[1:]
    # if dataset_type not in vaex.dataset.dataset_type_map:
    # print "unknown type", dataset_type
    # print "possible options are:\n\t", "\n\t".join(vaex.dataset.dataset_type_map.keys())
    # sys.exit(1)
    # evaluated_options = []
    # for option in dataset_options:
    # evaluated_options.append(eval(option))

    class_ = vaex.dataset.dataset_type_map[options.type]
    # @datasets = [class_(filename, *options) for filename in inputs]
    datasets_list = []
    for filename in inputs:
        print(("loading file", filename, options.particle_types))
        datasets = []
        for type in options.particle_types.split(","):
            datasets.append(class_(filename + "#" + type))
        datasets_list.append(datasets)
    datasets_centering = None
    if options.center_type:
        datasets_centering = []
        for filename in inputs:
            datasets_centering.append(class_(filename + "#" + options.center_type))

    merge(output, datasets_list, datasets_centering, options.sort, ascending=not options.reverse, order_column_name=options.order)
