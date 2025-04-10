# Introduction to `.secrets` and `.env`

- The `dot-` in the file/directory names refers to substituting them with `.` in your cloned files. This is used to help you get started with environment variables.
- In your cloned dir, copy the `dot-secrets` to `.secrets` and any `dot-env...` to `.env...`.

```bash
BASE_PATH="./../.secrets" && mkdir $BASE_PATH &&\
for dotfile in dot-*; do newname=".${dotfile#dot-}" && newname="${newname%.toml}" && cp "$dotfile" "$BASE_PATH/$newname"; done
```
