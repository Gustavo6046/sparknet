project('sparknet', 'c')

cc = meson.get_compiler('c')

depmath = cc.find_library('m', required: true)

libinc = include_directories('include')
libsrc = [
    'src/layer.c',
    'src/synapse.c',
    'src/network.c'
]
libdeps = [
    depmath
]

libsparknet = library('sparknet', libsrc, include_directories: libinc, dependencies: libdeps)

executable('sparknet_test', 'test.c', include_directories: libinc, dependencies: depmath, link_with: libsparknet)
