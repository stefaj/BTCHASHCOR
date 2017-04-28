with import <nixpkgs> {};
with pkgs.python27Packages;

buildPythonPackage{
    name = "numerai";
    buildInputs = [ python27Full
                    python27Packages.matplotlib
                    python27Packages.futures
                    python27Packages.future
                    python27Packages.twisted
                    python27Packages.scipy
                    python27Packages.setuptools
                    python27Packages.pandas
                    python27Packages.httplib2
                    python27Packages.urllib3
                    python27Packages.numpy
                    python27Packages.requests2
                    python27Packages.websocket_client
                   ]; 

}

