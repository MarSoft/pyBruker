# PyBruker
## Version: 1.1.2

The tools to convert Bruker raw to Nifti format.

List of known issues,
1. Incorrect orientation. (espetially for obliqued image.): currently, all oblique images are unwarped
2. DTI header information is not compatible at this point. (working on to integrate it to extension header)

### Requirements
- Linux or Mac OSX

### Installation
```angular2html
pip install pyBruker
```

### Command line tool
- Help function
```angular2html
brkraw -h
```

- Print out summary of the scan
```angular2html
brkraw summary <session path>
```

- Convert a whole session
```angular2html
brkraw tonii <session path>
```

- Convert only one scan in the session
```angular2html
brkraw tonii <session path> -s <scan id> -r <reco id>
```

- If <reco id> is not provided, then default is 1

- To convert all raw data under the folder. This command will scan all folder under the parent folder and the derived file will be structured as BIDS
```angular2html
brkraw tonii_all <parent folder>
```