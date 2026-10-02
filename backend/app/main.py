def health_check() -> dict[str, str]:
    """Return the current application health status."""
    return {"status": "healthy"}


if __name__ == "__main__":
    print(health_check())
