
def add_user(store: UserStore, name: str, display_name: str,
             is_human: bool = True) -> User:
    user = User(id=UserId(str(uuid.uuid4())), name=name,
                display_name=display_name, is_human=is_human)
    store.users.append(user)
    return user


def edit_user(store: UserStore, user_id: UserId, **changes) -> User:
    for user in store.users:
        if user.id == user_id:
            for key, value in changes.items():
                if not hasattr(user, key):
                    raise ValueError(f"Unknown user field: {key}")
                setattr(user, key, value)
            return user
    raise ValueError(f"Could not find user with id {user_id}")
