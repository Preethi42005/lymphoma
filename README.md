# Welcome to your Lovable project

This project was built with [Lovable](https://lovable.dev).

## Build with Lovable

Open your project in the [Lovable editor](https://lovable.dev) and keep building.

- **Ship faster**: describe what you want to build and Lovable handles the code.
- **Stay in sync**: connect the project to GitHub and every change made in Lovable is committed straight to your repository.
- **Full ownership**: this code is yours. Push to your repository and your changes sync back into Lovable, ready for your next prompt.

## Development

Prefer working locally? You need Node.js and npm — [install with nvm](https://github.com/nvm-sh/nvm#installing-and-updating).

```sh
git clone <this-repository-url>
cd <repository-name>
npm i
npm run dev
```

### Auth + Excel user storage

Sign-up and sign-in details are saved in `users.xlsx` (sheets: **Users**, **LoginHistory**).

1. Start the auth API (port 5002):

```sh
npm run auth
```

2. In another terminal, start the web app:

```sh
npm run dev
```

3. Open `/login` to create an account or sign in. New rows appear in `users.xlsx`.

Passwords are stored as SHA-256 hashes (not plain text). Close and reopen the Excel file after sign-up/login to see updates.

## Built with

- TanStack Start
- TypeScript
- React
- Tailwind CSS
- Flask auth API + pandas/openpyxl (`users.xlsx`)
