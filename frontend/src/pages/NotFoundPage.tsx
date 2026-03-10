import { Link } from 'react-router-dom';
import { Button } from '../components/ui/Button';

export default function NotFoundPage() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-background">
      <div className="text-center">
        <h1 className="text-6xl font-bold text-border mb-4">404</h1>
        <h2 className="text-xl font-semibold text-text-primary mb-2">Page Not Found</h2>
        <p className="text-text-muted mb-6">The page you're looking for doesn't exist.</p>
        <Link to="/">
          <Button>Back to Dashboard</Button>
        </Link>
      </div>
    </div>
  );
}
