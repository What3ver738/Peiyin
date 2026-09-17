# Local-use security

Peiyin is designed for one person on their own computer. The interface binds to
`127.0.0.1`, disables public sharing, monitoring, MCP access, run history and
Gradio analytics, and serializes settings/profile changes with rendering jobs.
Do not expose it through a tunnel, reverse proxy or public server. Loopback is
not isolation from other users or processes on the same machine.

Saved API keys stay out of the browser's initial configuration. Only an explicit
Settings save writes a newly entered key to disk. Environment variables override
entered and stored keys. Configuration is replaced atomically using a private
temporary file. On Windows, protect the user directory with the account's ACLs;
POSIX mode numbers do not describe Windows access permissions.

Only explicit result/preview copies are made available as downloads. The private
application directory is blocked from the file-serving route. Uploads and
browser download copies use temporary storage; finished files and work caches
remain on disk until the user removes them. Use only your own output folders.
Temporary copies consume additional space, particularly for batch jobs.

Run `python -m pip_audit` in the installed environment after dependency updates.
CI checks unit tests, dependency advisories, source hygiene and wheel contents.
A clean advisory result is a dated check, not a guarantee of security.

Do not post API keys, private video, transcripts, config files or unredacted logs
in issues. Rotate a key with its provider if it has been exposed. When reporting
a bug, include OS, Python version and a minimal reproduction using invented data.
