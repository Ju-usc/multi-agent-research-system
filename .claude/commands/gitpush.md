Analyze changes and intelligently add, commit, and push $ARGUMENTS:

1. Run git analysis commands (simple examples):
   - git status (untracked files, modifications)
   - git diff (unstaged changes)
   - git diff --staged (staged changes)
   - git log -1 --oneline (last commit for context)
2. Stage appropriate files based on analysis
3. Generate context-aware commit message from diff output
4. Commit and push to origin

Handle edge cases appropriately (conflicts, no upstream, etc.)
Make sure the style of the commit message is consistent with the rest of the codebase.
Make sure commit message is not too long and describes the changes in bullet points.