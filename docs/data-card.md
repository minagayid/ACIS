# Data card

## Status

The committed example corpus is synthetic metadata with hand-authored acoustic summaries. It is a software fixture, not an empirical animal dataset. Its metrics must not be generalized to dogs.

## Intended future data

Consented, species-specific recordings with synchronized audio/video where feasible, session metadata, observable behavior, trigger metadata, source identity confidence, population/region/habitat metadata, and negative/background windows.

## Required provenance

- species, subspecies, population, region, habitat, animal, session, social group, site, device, and observer identifiers;
- capture timestamps and synchronized sensor clocks;
- file hash, transformation history, and preprocessing version;
- license, consent, permits, and welfare review status;
- sensitive-location handling and access policy;
- independent annotations, disagreement, and adjudication history.

## Known limitations

Household recordings are not representative of all dogs. Breed, age, sex, health, training, owner behavior, room acoustics, device type, and recording schedule can become shortcuts. A large number of files does not substitute for sender identity, production context, receiver context, or independent behavioral outcomes.

## Governance requirements

Do not publish exact locations of vulnerable animals. Do not release recordings containing people or children without consent and privacy review. Do not use this system for veterinary triage, aggression decisions, wildlife management, or any safety-critical action without qualified human oversight.
