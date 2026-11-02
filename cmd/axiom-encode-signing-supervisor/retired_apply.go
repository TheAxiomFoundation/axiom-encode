//go:build darwin || linux

package main

import (
	"bytes"
	"encoding/json"
	"errors"
	"regexp"
	"strings"
	"unicode/utf8"
)

// This inventory comes only from the owner-provisioned protected trust file.
// It carries digests of the exact artifacts and covered files at an immutable
// pre-rotation cutoff; it never grants an additional signing capability.
type retiredApplyInventory struct {
	Schema       string                   `json:"schema"`
	Repositories []retiredApplyRepository `json:"repositories"`
}

type retiredApplyRepository struct {
	Repository   string                 `json:"repository"`
	CutoffCommit string                 `json:"cutoff_commit"`
	Artifacts    []retiredApplyArtifact `json:"artifacts"`
}

type retiredApplyArtifact struct {
	Path         string             `json:"path"`
	SHA256       string             `json:"sha256"`
	CoveredFiles []retiredApplyFile `json:"covered_files"`
}

type retiredApplyFile struct {
	Path         string  `json:"path"`
	SHA256       string  `json:"sha256"`
	SourceCommit *string `json:"source_commit,omitempty"`
}

var retiredRepositoryPattern = regexp.MustCompile(`^TheAxiomFoundation/rulespec-[a-z0-9]+(-[a-z0-9]+)*$`)
var retiredSHA256Pattern = regexp.MustCompile(`^[0-9a-f]{64}$`)
var retiredCommitPattern = regexp.MustCompile(`^[0-9a-f]{40}$`)

func validRetiredArtifactPath(path string) bool {
	if path == "" || !utf8.ValidString(path) || strings.ContainsAny(path, "\\*?[") || strings.HasPrefix(path, "/") || strings.HasPrefix(path, "-") || strings.HasPrefix(path, ":") {
		return false
	}
	for _, character := range path {
		if character < 0x20 || (character >= 0x7f && character <= 0x9f) {
			return false
		}
	}
	for _, component := range strings.Split(path, "/") {
		if component == "" || component == "." || component == ".." {
			return false
		}
	}
	return true
}

func validateRetiredApplyInventory(inventory *retiredApplyInventory) error {
	if inventory == nil || inventory.Schema != "axiom-encode/retired-apply-inventory/v1" || len(inventory.Repositories) == 0 {
		return errors.New("retired apply inventory requires its v1 schema and repositories")
	}
	repositories := make(map[string]struct{})
	for _, repository := range inventory.Repositories {
		if !retiredRepositoryPattern.MatchString(repository.Repository) || !retiredCommitPattern.MatchString(repository.CutoffCommit) || len(repository.Artifacts) == 0 {
			return errors.New("retired apply inventory has an invalid repository, cutoff, or artifacts")
		}
		if _, duplicate := repositories[repository.Repository]; duplicate {
			return errors.New("retired apply inventory has duplicate repositories")
		}
		repositories[repository.Repository] = struct{}{}
		artifacts := make(map[string]struct{})
		for _, artifact := range repository.Artifacts {
			if !validRetiredArtifactPath(artifact.Path) || !retiredSHA256Pattern.MatchString(artifact.SHA256) || artifact.CoveredFiles == nil {
				return errors.New("retired apply inventory has an invalid artifact")
			}
			if _, duplicate := artifacts[artifact.Path]; duplicate {
				return errors.New("retired apply inventory has duplicate artifact paths")
			}
			artifacts[artifact.Path] = struct{}{}
			covered := map[string]struct{}{artifact.Path + "\x00" + repository.CutoffCommit: {}}
			for _, file := range artifact.CoveredFiles {
				if file.Path == artifact.Path || !validRetiredArtifactPath(file.Path) || !retiredSHA256Pattern.MatchString(file.SHA256) {
					return errors.New("retired apply inventory has an invalid covered file")
				}
				source := repository.CutoffCommit
				if file.SourceCommit != nil {
					if !retiredCommitPattern.MatchString(*file.SourceCommit) {
						return errors.New("retired apply inventory has an invalid covered source commit")
					}
					source = *file.SourceCommit
				}
				identity := file.Path + "\x00" + source
				if _, duplicate := covered[identity]; duplicate {
					return errors.New("retired apply inventory has duplicate covered file paths")
				}
				covered[identity] = struct{}{}
			}
		}
	}
	return nil
}

func (file *retiredApplyFile) UnmarshalJSON(raw []byte) error {
	if err := rejectRetiredInventoryFields(raw, []string{"path", "sha256"}, "source_commit"); err != nil {
		return err
	}
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	if fields == nil {
		return errors.New("covered file must be an object")
	}
	for name := range fields {
		if name != "path" && name != "sha256" && name != "source_commit" {
			return errors.New("covered file has unknown fields")
		}
	}
	if source, present := fields["source_commit"]; present && bytes.Equal(bytes.TrimSpace(source), []byte("null")) {
		return errors.New("covered source commit must be a full commit SHA")
	}
	type wireFile retiredApplyFile
	var decoded wireFile
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*file = retiredApplyFile(decoded)
	return nil
}

func rejectRetiredInventoryFields(raw []byte, required []string, optional ...string) error {
	var fields map[string]json.RawMessage
	if err := json.Unmarshal(raw, &fields); err != nil {
		return err
	}
	if fields == nil {
		return errors.New("retired apply inventory entries must be objects")
	}
	allowed := make(map[string]struct{}, len(required)+len(optional))
	for _, name := range required {
		allowed[name] = struct{}{}
		if _, present := fields[name]; !present {
			return errors.New("retired apply inventory entry lacks required fields")
		}
	}
	for _, name := range optional {
		allowed[name] = struct{}{}
	}
	for name := range fields {
		if _, present := allowed[name]; !present {
			return errors.New("retired apply inventory entry has unknown fields")
		}
	}
	return nil
}

func (inventory *retiredApplyInventory) UnmarshalJSON(raw []byte) error {
	if err := rejectRetiredInventoryFields(raw, []string{"schema", "repositories"}); err != nil {
		return err
	}
	type wireInventory retiredApplyInventory
	var decoded wireInventory
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*inventory = retiredApplyInventory(decoded)
	return nil
}

func (repository *retiredApplyRepository) UnmarshalJSON(raw []byte) error {
	if err := rejectRetiredInventoryFields(raw, []string{"repository", "cutoff_commit", "artifacts"}); err != nil {
		return err
	}
	type wireRepository retiredApplyRepository
	var decoded wireRepository
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*repository = retiredApplyRepository(decoded)
	return nil
}

func (artifact *retiredApplyArtifact) UnmarshalJSON(raw []byte) error {
	if err := rejectRetiredInventoryFields(raw, []string{"path", "sha256", "covered_files"}); err != nil {
		return err
	}
	type wireArtifact retiredApplyArtifact
	var decoded wireArtifact
	if err := json.Unmarshal(raw, &decoded); err != nil {
		return err
	}
	*artifact = retiredApplyArtifact(decoded)
	return nil
}
